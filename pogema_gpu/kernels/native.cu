// Native Pogema 1.3.2a4 soft collisions, plus optional DMM observation support.
// Integer arithmetic only.
#include <torch/extension.h>
#include <climits>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace {
__device__ int move_delta(int action, int width) {
  return action == 1 ? -width : action == 2 ? width : action == 3 ? -1 : action == 4 ? 1 : 0;
}

// A cell owns an ordered doubly linked list, just like native used_cells[cell].
__device__ void remove_agent(int i, int cell, int* head, int* tail, int* next, int* prev) {
  if (prev[i] < 0) head[cell] = next[i]; else next[prev[i]] = next[i];
  if (next[i] < 0) tail[cell] = prev[i]; else prev[next[i]] = prev[i];
}
__device__ void append_agent(int i, int cell, int* head, int* tail, int* next, int* prev) {
  prev[i] = tail[cell]; next[i] = -1;
  if (tail[cell] < 0) head[cell] = i; else next[tail[cell]] = i;
  tail[cell] = i;
}

// One sequential state machine per independent environment. No recursive CUDA
// calls, convergence readback, arbitrary iteration limit, or inter-block races.
__global__ void resolve_soft_kernel(const bool* walls, const int64_t* positions,
    const int64_t* proposed, int64_t* executed, int64_t* next_positions, const bool* finished, int* scratch,
    int n, int cells, int width) {
  if (threadIdx.x != 0) return;
  int e = blockIdx.x;
  if (finished[e]) return;
  positions += e*n; proposed += e*n; executed += e*n; next_positions += e*n;
  walls += e*cells;
  scratch += e*(3*cells + 4*n);
  int* head = scratch; int* tail = head+cells; int* occupied = tail+cells;
  int* dest = occupied+cells; int* next = dest+n; int* prev = next+n; int* raw = prev+n;
  for (int c=0; c<cells; ++c) head[c] = tail[c] = occupied[c] = -1;
  for (int i=0; i<n; ++i) {
    // Python's public boundary validates action range asynchronously on device.
    executed[i] = proposed[i];
    dest[i] = raw[i] = positions[i] + move_delta(proposed[i], width);
    occupied[positions[i]] = i;
    append_agent(i, dest[i], head, tail, next, prev);
  }
  // Native used_edges overwrites the later agent's reverse entry. Cancel only
  // the lower-ID endpoint here; native's following cascade cancels its peer.
  for (int i=0; i<n; ++i) {
    int j = occupied[raw[i]];
    if (j > i && raw[j] == positions[i]) {
      remove_agent(i, dest[i], head, tail, next, prev);
      dest[i] = positions[i]; executed[i] = 0;
      append_agent(i, dest[i], head, tail, next, prev);
    }
  }
  for (int root=n-1; root>=0; --root) {
    int cell = dest[root];
    if (head[cell] == tail[cell] && !walls[cell]) continue;
    int i = root;
    while (true) {
      remove_agent(i, dest[i], head, tail, next, prev);
      dest[i] = positions[i]; executed[i] = 0;
      int previous_head = head[dest[i]];
      append_agent(i, dest[i], head, tail, next, prev);
      if (previous_head < 0) break;
      i = previous_head;
    }
  }
  for (int i=0; i<n; ++i) next_positions[i] = dest[i];
}

// Shared commit for native-soft and collision-free planner outputs. No map-sized
// scratch, conflict resolution, or planner-specific state belongs here.
__global__ void commit_kernel(int64_t* positions, const int64_t* goals,
    const int64_t* next_positions, const int64_t* submitted, const int64_t* actions,
    int64_t* executed, int64_t* steps, const int64_t* horizons, bool* finished,
    bool* solved, int64_t* repairs, int64_t* solve_time, int n) {
  int e = blockIdx.x, t = threadIdx.x;
  if (finished[e]) return; // uniform across the block
  int64_t base = int64_t(e)*n;
  __shared__ int off_goal[256], changed[256];
  __shared__ bool terminal;
  __shared__ int64_t step;
  if (t == 0) step = steps[e];
  int misses = 0, fixes = 0;
  for (int i=t; i<n; i+=blockDim.x) {
    int64_t k = base+i;
    positions[k] = next_positions[k];
    executed[k] = actions[k];
    misses += next_positions[k] != goals[k];
    fixes += submitted[k] != actions[k];
  }
  off_goal[t] = misses; changed[t] = fixes;
  __syncthreads();
  for (int stride=128; stride>0; stride/=2) {
    if (t < stride) { off_goal[t] += off_goal[t+stride]; changed[t] += changed[t+stride]; }
    __syncthreads();
  }
  if (t == 0) {
    steps[e] = step+1;
    solved[e] = off_goal[0] == 0;
    terminal = solved[e] || steps[e] >= horizons[e];
    finished[e] = terminal;
    repairs[e] += changed[0];
  }
  __syncthreads();
  for (int i=t; i<n; i+=blockDim.x) {
    int64_t k = base+i;
    bool on_goal = positions[k] == goals[k];
    if (solve_time[k] < 0 && (on_goal || terminal)) solve_time[k] = step;
    // Preserve native's final-step goal-departure quirk.
    if (!on_goal && !terminal) solve_time[k] = -1;
  }
}

// Static goal BFS: each cell is enqueued at most once, so cells-sized queues
// are sufficient. Python chunks scratch storage and enforces a cache budget.
__global__ void bfs_kernel(const bool* walls, const int64_t* goals,
    const int64_t* slots, int32_t* distances, int32_t* queues,
    int n, int cells, int width, int count, int offset, bool* overflow) {
  if (threadIdx.x != 0 || blockIdx.x >= count) return;
  int item = offset + blockIdx.x, e = slots[item/n], a = item%n;
  walls += int64_t(e)*cells;
  int32_t* d = distances + (int64_t(e)*n+a)*cells;
  int32_t* queue = queues + int64_t(blockIdx.x)*cells;
  for (int c=0; c<cells; ++c) d[c] = -1;
  int goal = goals[e*n+a], tail=1;
  queue[0] = goal; d[goal] = 0;
  for (int head=0; head<tail; ++head) {
    int cell=queue[head];
    for (int action=1; action<5; ++action) {
      int target=cell+move_delta(action,width);
      if (target < 0 || target >= cells ||
          (action==3 && cell%width==0) || (action==4 && cell%width==width-1)) continue;
      if (!walls[target] && d[target]<0) {
        d[target]=d[cell]+1; queue[tail++]=target;
        if (overflow && d[target]>=65535) overflow[blockIdx.x]=true;
      }
    }
  }
}

// Stream full-map BFS through bounded scratch, retaining only a 129x129
// grid-step=64 window per agent. No agents*full-map allocation/index exists.
__device__ void window_bfs_one(const bool* walls, const int64_t* goals,
    const int64_t* positions, int item, int32_t* windows,
    int32_t* origins, int32_t* scratch, int32_t* queues, bool* overflow,
    int n, int cells, int width) {
  int e = item/n;
  walls += int64_t(e)*cells;
  int32_t* d = scratch + int64_t(blockIdx.x)*cells;
  int32_t* queue = queues + int64_t(blockIdx.x)*cells;
  for (int c=0; c<cells; ++c) d[c] = -1;
  int goal = goals[item], tail=1;
  queue[0]=goal; d[goal]=0;
  for (int head=0; head<tail; ++head) {
    int cell=queue[head];
    for (int action=1; action<5; ++action) {
      int target=cell+move_delta(action,width);
      if (target<0 || target>=cells || (action==3 && cell%width==0) ||
          (action==4 && cell%width==width-1)) continue;
      if (!walls[target] && d[target]<0) {
        d[target]=d[cell]+1; queue[tail++]=target;
        // The tokenizer reserves uint16 value 65535 for unreachable cells.
        if (d[target]>=65535) overflow[blockIdx.x]=true;
      }
    }
  }
  int pos=positions[item], row=max(pos/width-5,0)/64*64, col=max(pos%width-5,0)/64*64;
  origins[item*2]=row; origins[item*2+1]=col;
  int32_t* out=windows+int64_t(item)*129*129;
  for (int r=0; r<129; ++r) for (int c=0; c<129; ++c)
    out[r*129+c]=(row+r<cells/width && col+c<width) ? d[(row+r)*width+col+c] : -1;
}

__global__ void window_bfs_kernel(const bool* walls, const int64_t* goals,
    const int64_t* positions, const int64_t* agents, int32_t* windows,
    int32_t* origins, int32_t* scratch, int32_t* queues, bool* overflow,
    int n, int cells, int width, int count, int offset) {
  if (threadIdx.x != 0 || blockIdx.x >= count) return;
  window_bfs_one(walls,goals,positions,agents[offset+blockIdx.x],windows,
      origins,scratch,queues,overflow,n,cells,width);
}

// Bounded scratch lanes consume a device work queue. Neither the number of
// stale windows nor their indices need to cross to the host.
__global__ void window_bfs_dispatch_kernel(const bool* walls, const int64_t* goals,
    const int64_t* positions, const bool* finished, const int64_t* shapes,
    int32_t* windows, int32_t* origins, int32_t* scratch, int32_t* queues,
    bool* overflow, int32_t* work, int n, int total, int cells, int width) {
  while (true) {
    int item=atomicAdd(work,1);
    if (item>=total) return;
    int e=item/n, top=origins[item*2], left=origins[item*2+1];
    int row=positions[item]/width, col=positions[item]%width;
    int bottom=min(top+128,int(shapes[e*2]-1));
    int right=min(left+128,int(shapes[e*2+1]-1));
    bool stale=top<0 || (!finished[e] &&
        (row-5<top || row+5>bottom || col-5<left || col+5>right));
    if (!stale) continue;
    atomicAdd(work+1,1);
    window_bfs_one(walls,goals,positions,item,windows,origins,scratch,queues,
        overflow,n,cells,width);
  }
}

__device__ int distance_at(const int32_t* d, int pos, int width, const int32_t* origin) {
  return origin ? d[(pos/width-origin[0])*129+pos%width-origin[1]] : d[pos];
}
__device__ int hint(const int32_t* d, int pos, int width, const int32_t* origin) {
  int value=distance_at(d,pos,width,origin), center=value<0 ? 65535 : value, bits=0;
  for (int a=1; a<5; ++a) {
    int value=distance_at(d,pos+move_delta(a,width),width,origin);
    bits=bits*2+(value>=0 && value<center);
  }
  return 50+bits;
}
__global__ void tokens_kernel(const int64_t* positions, const int64_t* goals,
    const int32_t* distances, const int64_t* history, const int64_t* slots,
    int64_t* tokens, int64_t* neighbors, int n, int cells, int width, int total,
    const int32_t* origins) {
  int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index>=total) return;
  int e=slots[index/n], agent=index%n;
  positions += e*n; goals += e*n; history += e*n*5;
  const int32_t* d=distances+int64_t(e*n+agent)*cells;
  const int32_t* origin=origins ? origins+(e*n+agent)*2 : nullptr;
  int64_t* out=tokens+index*256;
  int64_t* chat=neighbors+index*13;
  int pos=positions[agent], row=pos/width, col=pos%width;
  int value=distance_at(d,pos,width,origin), center=value<0 ? 65535 : value, t=0;
  for (int dr=-5; dr<=5; ++dr) for (int dc=-5; dc<=5; ++dc) {
    int value=distance_at(d,pos+dr*width+dc,width,origin), delta=value-center;
    out[t++]=value<0 ? 41 : delta < -20 ? 42 : delta > 20 ? 43 : delta+20;
  }
  // Small DMM cohorts: a readable stable top-13 scan. No dense NxN tensor.
  int ids[13], keys[13], count=0;
  for (int j=0; j<n; ++j) {
    int dr=positions[j]/width-row, dc=positions[j]%width-col;
    if (abs(dr)>5 || abs(dc)>5) continue;
    int key=(abs(dr)+abs(dc))*n+j, k=count<13 ? count++ : 13;
    while (k>0 && key<keys[k-1]) { if (k<13) { keys[k]=keys[k-1]; ids[k]=ids[k-1]; } --k; }
    if (k<13) { keys[k]=key; ids[k]=j; }
  }
  for (int k=0; k<13; ++k) {
    chat[k]=k<count ? ids[k] : -1;
    if (k>=count) continue;
    int j=ids[k];
    out[t++]=positions[j]/width-row+20; out[t++]=positions[j]%width-col+20;
    out[t++]=max(-20,min(20,int(goals[j]/width-row)))+20;
    out[t++]=max(-20,min(20,int(goals[j]%width-col)))+20;
    for (int h=0; h<5; ++h) out[t++]=history[j*5+h];
    out[t++]=hint(distances+int64_t(e*n+j)*cells, positions[j], width,
                  origins ? origins+(e*n+j)*2 : nullptr);
  }
  while (t<256) out[t++]=66;
}

void check(const torch::Tensor& t, torch::ScalarType dtype, const torch::Tensor& reference) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type()==dtype &&
              t.device()==reference.device(), "expected contiguous same-device CUDA tensor with correct dtype");
}
} // namespace

void resolve_soft(torch::Tensor walls, torch::Tensor positions, torch::Tensor proposed,
                  torch::Tensor executed, torch::Tensor next_positions,
                  torch::Tensor finished, torch::Tensor scratch) {
  for (auto t : {positions, proposed, executed, next_positions}) check(t,torch::kInt64,walls);
  for (auto t : {walls, finished}) check(t,torch::kBool,walls);
  check(scratch,torch::kInt32,walls);
  TORCH_CHECK(walls.dim()==3 && positions.dim()==2, "invalid state rank");
  int b=positions.size(0), n=positions.size(1), cells=walls.size(1)*walls.size(2);
  TORCH_CHECK(b>0 && n>0 && walls.size(0)==b, "invalid batch shape");
  for (auto t : {proposed,executed,next_positions}) TORCH_CHECK(t.sizes()==positions.sizes(), "agent shape mismatch");
  TORCH_CHECK(finished.dim()==1 && finished.numel()==b, "environment shape mismatch");
  TORCH_CHECK(scratch.numel()>=b*(3LL*cells+4LL*n), "scratch too small");
  c10::cuda::CUDAGuard guard(walls.device());
  resolve_soft_kernel<<<b,1,0,c10::cuda::getCurrentCUDAStream()>>>(walls.data_ptr<bool>(),positions.data_ptr<int64_t>(),
    proposed.data_ptr<int64_t>(),executed.data_ptr<int64_t>(),next_positions.data_ptr<int64_t>(),finished.data_ptr<bool>(),
    scratch.data_ptr<int>(),n,cells,walls.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void commit(torch::Tensor positions, torch::Tensor goals, torch::Tensor next_positions,
            torch::Tensor submitted, torch::Tensor actions, torch::Tensor executed,
            torch::Tensor steps, torch::Tensor horizons, torch::Tensor finished,
            torch::Tensor solved, torch::Tensor repairs, torch::Tensor solve_time) {
  for (auto t : {positions, goals, next_positions, submitted, actions, executed, steps, horizons, repairs, solve_time})
    check(t,torch::kInt64,positions);
  for (auto t : {finished, solved}) check(t,torch::kBool,positions);
  TORCH_CHECK(positions.dim()==2 && positions.size(0)>0 && positions.size(1)>0, "invalid positions");
  int b=positions.size(0), n=positions.size(1);
  for (auto t : {goals,next_positions,submitted,actions,executed,solve_time})
    TORCH_CHECK(t.sizes()==positions.sizes(), "agent shape mismatch");
  for (auto t : {steps,horizons,finished,solved,repairs})
    TORCH_CHECK(t.dim()==1 && t.numel()==b, "environment shape mismatch");
  c10::cuda::CUDAGuard guard(positions.device());
  commit_kernel<<<b,256,0,c10::cuda::getCurrentCUDAStream()>>>(positions.data_ptr<int64_t>(),goals.data_ptr<int64_t>(),
    next_positions.data_ptr<int64_t>(),submitted.data_ptr<int64_t>(),actions.data_ptr<int64_t>(),executed.data_ptr<int64_t>(),
    steps.data_ptr<int64_t>(),horizons.data_ptr<int64_t>(),finished.data_ptr<bool>(),solved.data_ptr<bool>(),
    repairs.data_ptr<int64_t>(),solve_time.data_ptr<int64_t>(),n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Compatibility entry point; both paths use exactly the same commit kernel.
void step(torch::Tensor walls, torch::Tensor positions, torch::Tensor goals,
          torch::Tensor proposed, torch::Tensor executed, torch::Tensor steps, torch::Tensor horizons,
          torch::Tensor finished, torch::Tensor solved, torch::Tensor repairs,
          torch::Tensor solve_time, torch::Tensor scratch) {
  auto next_positions = torch::empty_like(positions), actions = torch::empty_like(executed);
  resolve_soft(walls,positions,proposed,actions,next_positions,finished,scratch);
  commit(positions,goals,next_positions,proposed,actions,executed,steps,horizons,finished,solved,repairs,solve_time);
}

void bfs(torch::Tensor walls, torch::Tensor goals, torch::Tensor slots,
         torch::Tensor distances, torch::Tensor queues, int64_t offset, int64_t count,
         c10::optional<torch::Tensor> overflow = c10::nullopt) {
  check(walls,torch::kBool,walls);
  for (auto t : {goals,slots}) check(t,torch::kInt64,walls);
  for (auto t : {distances,queues}) check(t,torch::kInt32,walls);
  TORCH_CHECK(walls.dim()==3 && goals.dim()==2 && slots.dim()==1, "invalid BFS shape");
  int b=walls.size(0), n=goals.size(1), cells=walls.size(1)*walls.size(2);
  TORCH_CHECK(goals.size(0)==b && distances.numel()==b*int64_t(n)*cells &&
              count>0 && offset>=0 && offset+count<=slots.numel()*n && queues.numel()>=count*cells, "invalid BFS buffers");
  if (overflow.has_value()) {
    check(*overflow,torch::kBool,walls);
    TORCH_CHECK(overflow->numel()>=count, "overflow flags too small");
  }
  c10::cuda::CUDAGuard guard(walls.device());
  bfs_kernel<<<count,1,0,c10::cuda::getCurrentCUDAStream()>>>(walls.data_ptr<bool>(),goals.data_ptr<int64_t>(),
    slots.data_ptr<int64_t>(),distances.data_ptr<int32_t>(),queues.data_ptr<int32_t>(),n,cells,walls.size(2),count,offset,
    overflow.has_value() ? overflow->data_ptr<bool>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> tokens(torch::Tensor positions, torch::Tensor goals,
    torch::Tensor distances, torch::Tensor history, torch::Tensor slots, int64_t width,
    c10::optional<torch::Tensor> origins = c10::nullopt) {
  for (auto t : {positions,goals,history,slots}) check(t,torch::kInt64,positions);
  check(distances,torch::kInt32,positions);
  TORCH_CHECK(positions.dim()==2 && goals.sizes()==positions.sizes() && distances.dim()==3 && slots.dim()==1, "invalid token shape");
  int b=positions.size(0), n=positions.size(1), count=slots.numel(), cells=distances.size(2);
  TORCH_CHECK(width>10 && distances.size(0)==b && distances.size(1)==n && history.numel()==b*n*5LL, "invalid token buffers");
  if (origins.has_value()) {
    check(*origins,torch::kInt32,positions);
    TORCH_CHECK(origins->sizes()==torch::IntArrayRef({b,n,2}) && cells==129*129, "invalid window origins");
  } else TORCH_CHECK(cells%width==0, "invalid full-map distance shape");
  c10::cuda::CUDAGuard guard(positions.device());
  auto out=torch::empty({count,n,256},positions.options());
  auto chat=torch::empty({count,n,13},positions.options());
  if (count>0) tokens_kernel<<<(count*n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    positions.data_ptr<int64_t>(),goals.data_ptr<int64_t>(),distances.data_ptr<int32_t>(),history.data_ptr<int64_t>(),
    slots.data_ptr<int64_t>(),out.data_ptr<int64_t>(),chat.data_ptr<int64_t>(),n,cells,width,count*n,
    origins.has_value() ? origins->data_ptr<int32_t>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out,chat};
}
void window_bfs(torch::Tensor walls, torch::Tensor goals, torch::Tensor positions,
    torch::Tensor agents, torch::Tensor windows, torch::Tensor origins,
    torch::Tensor scratch, torch::Tensor queues, torch::Tensor overflow, int64_t offset, int64_t count) {
  check(walls,torch::kBool,walls); check(overflow,torch::kBool,walls);
  for (auto t : {goals,positions,agents}) check(t,torch::kInt64,walls);
  for (auto t : {windows,origins,scratch,queues}) check(t,torch::kInt32,walls);
  TORCH_CHECK(walls.dim()==3 && goals.dim()==2 && positions.sizes()==goals.sizes() && agents.dim()==1, "invalid window BFS shape");
  int64_t b=goals.size(0), n=goals.size(1), cells=walls.size(1)*walls.size(2);
  TORCH_CHECK(walls.size(0)==b && windows.sizes()==torch::IntArrayRef({b,n,129*129}) &&
              origins.sizes()==torch::IntArrayRef({b,n,2}) && count>0 && offset>=0 && offset+count<=agents.numel() &&
              scratch.numel()>=count*cells && queues.numel()>=count*cells && overflow.numel()>=count, "invalid window BFS buffers");
  c10::cuda::CUDAGuard guard(walls.device());
  window_bfs_kernel<<<count,1,0,c10::cuda::getCurrentCUDAStream()>>>(walls.data_ptr<bool>(),goals.data_ptr<int64_t>(),
    positions.data_ptr<int64_t>(),agents.data_ptr<int64_t>(),windows.data_ptr<int32_t>(),origins.data_ptr<int32_t>(),
    scratch.data_ptr<int32_t>(),queues.data_ptr<int32_t>(),overflow.data_ptr<bool>(),n,cells,walls.size(2),count,offset);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void window_bfs_dispatch(torch::Tensor walls, torch::Tensor goals, torch::Tensor positions,
    torch::Tensor finished, torch::Tensor shapes, torch::Tensor windows, torch::Tensor origins,
    torch::Tensor scratch, torch::Tensor queues, torch::Tensor overflow, torch::Tensor work) {
  check(walls,torch::kBool,walls); check(finished,torch::kBool,walls); check(overflow,torch::kBool,walls);
  for (auto t : {goals,positions,shapes}) check(t,torch::kInt64,walls);
  for (auto t : {windows,origins,scratch,queues,work}) check(t,torch::kInt32,walls);
  TORCH_CHECK(walls.dim()==3 && goals.dim()==2 && positions.sizes()==goals.sizes(),"invalid dispatch shapes");
  int64_t b=goals.size(0), n=goals.size(1), cells=walls.size(1)*walls.size(2);
  TORCH_CHECK(b>0 && n>0 && b*n<INT_MAX && cells<INT_MAX && walls.size(0)==b &&
      finished.sizes()==torch::IntArrayRef({b}) && shapes.sizes()==torch::IntArrayRef({b,2}) &&
      windows.sizes()==torch::IntArrayRef({b,n,129*129}) && origins.sizes()==torch::IntArrayRef({b,n,2}) &&
      scratch.dim()==2 && scratch.size(0)>0 && scratch.size(0)<=b*n && scratch.size(1)==cells &&
      queues.sizes()==scratch.sizes() && overflow.numel()==scratch.size(0) && work.numel()==2,
      "invalid dispatch buffers");
  c10::cuda::CUDAGuard guard(walls.device());
  work.zero_(); overflow.zero_();
  window_bfs_dispatch_kernel<<<scratch.size(0),1,0,c10::cuda::getCurrentCUDAStream()>>>(
      walls.data_ptr<bool>(),goals.data_ptr<int64_t>(),positions.data_ptr<int64_t>(),
      finished.data_ptr<bool>(),shapes.data_ptr<int64_t>(),windows.data_ptr<int32_t>(),
      origins.data_ptr<int32_t>(),scratch.data_ptr<int32_t>(),queues.data_ptr<int32_t>(),
      overflow.data_ptr<bool>(),work.data_ptr<int32_t>(),n,b*n,cells,walls.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("window_bfs_dispatch", &window_bfs_dispatch);
  m.def("step", &step); m.def("window_bfs", &window_bfs);
  m.def("bfs", &bfs, pybind11::arg("walls"), pybind11::arg("goals"), pybind11::arg("slots"),
        pybind11::arg("distances"), pybind11::arg("queues"), pybind11::arg("offset"), pybind11::arg("count"),
        pybind11::arg("overflow")=c10::nullopt);
  m.def("resolve_soft", &resolve_soft); m.def("commit", &commit);
  m.def("tokens", &tokens, pybind11::arg("positions"), pybind11::arg("goals"), pybind11::arg("distances"),
        pybind11::arg("history"), pybind11::arg("slots"), pybind11::arg("width"), pybind11::arg("origins")=c10::nullopt);
}
