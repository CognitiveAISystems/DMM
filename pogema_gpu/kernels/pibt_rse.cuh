// Device-side bounded RSE. Included after PIBT and MT19937 device helpers.
// One block per environment: ordered RNG/PIBT, parallel ranking/history.
constexpr int kRSEThreads=128;
__global__ void pibt_rse_kernel(
    const int64_t* current, const int64_t* goals, const int64_t* candidates,
    const float* scores, const bool* valid, const int64_t* order,
    const int64_t* cells, const bool* active, const int32_t* history,
    const int64_t* hashes, const int64_t* coefficients, const int64_t* lengths,
    int64_t* rng, int64_t* occupied_now, int64_t* occupied_next,
    int64_t* next, int64_t* actions, int64_t* stack, int64_t* offsets,
    int64_t* preferences, bool* allowed, bool* forbidden, float* ties,
    int64_t* best_next, int64_t* best_actions, int64_t* stats,
    int64_t b, int64_t n, int64_t max_cells, int64_t horizon,
    int64_t limit, bool native_ties) {
  const int64_t e=blockIdx.x, base=e*n, abase=base*5, cbase=e*max_cells;
  const int lane=threadIdx.x;
  if (!active[e]) {
    for (int64_t i=lane; i<n; i+=blockDim.x) { next[base+i]=current[base+i]; actions[base+i]=0; }
    return;
  }
  __shared__ uint64_t partial_hash[kRSEThreads];
  __shared__ int64_t local_rng[625];
  __shared__ int repeated, detected, best_moves, save_best, retry;
  __shared__ int64_t escape, retries;
  if (native_ties) for (int i=lane; i<625; i+=blockDim.x) local_rng[i]=rng[e*625+i];
  if (lane==0) {
    escape=-1; detected=0; best_moves=0; retries=0;
    for (int64_t rank=0; rank<n; ++rank) {
      int64_t i=order[base+rank];
      if (current[base+i]!=goals[base+i]) { escape=i; break; }
    }
  }
  __syncthreads();
  for (int64_t attempt=0;; ++attempt) {
    if (attempt) for (int64_t i=lane; i<n; i+=blockDim.x)
      occupied_next[cbase+next[base+i]]=-1;
    // Exact native draw sequence is inherently ordered. Ranking is not.
    if (lane==0 && native_ties) {
      const int graph[5]={3,4,2,1,0};
      for (int64_t i=0; i<n; ++i) for (int k=0; k<5; ++k) {
        int64_t ix=abase+i*5+graph[k];
        float v=0;
        if (valid[ix]) {
          v=static_cast<float>(native_mt_next(local_rng))*0x1p-32f;
          if (v>=1.0f) v=0x1.fffffep-1f;
        }
        ties[ix]=v;
      }
    }
    __syncthreads();
    for (int64_t i=lane; i<n; i+=blockDim.x) {
      next[base+i]=-1; actions[base+i]=0;
      int pref[5]; float key[5], tie[5];
      const int graph[5]={3,4,2,1,0};
      for (int a=0; a<5; ++a) {
        int64_t ix=abase+i*5+a;
        allowed[ix]=valid[ix] && !forbidden[ix];
        key[a]=allowed[ix]?scores[ix]:-INFINITY;
        tie[a]=native_ties?ties[ix]:0;
        pref[a]=native_ties?graph[a]:a;
      }
      for (int j=1; j<5; ++j) {
        int a=pref[j], k=j;
        while (k>0 && (key[a]>key[pref[k-1]] ||
               (key[a]==key[pref[k-1]] && tie[a]<tie[pref[k-1]]))) {
          pref[k]=pref[k-1]; --k;
        }
        pref[k]=a;
      }
      for (int k=0; k<5; ++k) preferences[abase+i*5+k]=pref[k];
    }
    __syncthreads();
    if (lane==0) {
      pibt_environment(current,candidates,preferences,allowed,order,cells,
          occupied_now,occupied_next,next,actions,stack,offsets,b,n,max_cells,
          attempt==0,e);
      repeated=0;
    }
    __syncthreads();
    uint64_t hash=0;
    for (int64_t i=lane; i<n; i+=blockDim.x)
      hash+=static_cast<uint64_t>(next[base+i])*static_cast<uint64_t>(coefficients[i]);
    partial_hash[lane]=hash;
    __syncthreads();
    for (int offset=kRSEThreads/2; offset; offset/=2) {
      if (lane<offset) partial_hash[lane]+=partial_hash[lane+offset];
      __syncthreads();
    }
    // Parallel fingerprint search; only equal fingerprints load full states.
    for (int64_t t=lane; t<lengths[e]; t+=blockDim.x) {
      if (static_cast<uint64_t>(hashes[e*horizon+t])!=partial_hash[0]) continue;
      bool equal=true;
      for (int64_t i=0; i<n; ++i)
        if (next[base+i]!=history[(e*horizon+t)*n+i]) { equal=false; break; }
      if (equal) atomicExch(&repeated,1);
    }
    __syncthreads();
    if (lane==0) {
      bool moved=escape>=0 && next[base+escape]!=current[base+escape];
      save_best=attempt==0 || (repeated && moved && !best_moves);
      if (attempt==0) { detected=repeated; best_moves=moved; }
      else if (save_best) best_moves=1;
      retry=0;
      if (repeated && attempt<limit) {
        for (int64_t rank=0; rank<n; ++rank) {
          int64_t i=order[base+rank], ix=abase+i*5;
          if (current[base+i]==goals[base+i] || forbidden[ix+actions[base+i]]) continue;
          int available=0;
          for (int a=0; a<5; ++a) available+=valid[ix+a] && !forbidden[ix+a];
          if (available>1) {
            forbidden[ix+actions[base+i]]=true;
            retry=1; ++retries; break;
          }
        }
      }
    }
    __syncthreads();
    if (save_best) for (int64_t i=lane; i<n; i+=blockDim.x) {
      best_next[base+i]=next[base+i]; best_actions[base+i]=actions[base+i];
    }
    __syncthreads();
    if (!retry) break;
  }
  if (repeated) for (int64_t i=lane; i<n; i+=blockDim.x) {
    next[base+i]=best_next[base+i]; actions[base+i]=best_actions[base+i];
  }
  if (native_ties) for (int i=lane; i<625; i+=blockDim.x) rng[e*625+i]=local_rng[i];
  if (lane==0) {
    stats[e*4]=detected; stats[e*4+1]=retries;
    stats[e*4+2]=detected && !repeated; stats[e*4+3]=repeated;
  }
}

std::vector<torch::Tensor> pibt_rse(
    torch::Tensor current, torch::Tensor goals, torch::Tensor candidates,
    torch::Tensor scores, torch::Tensor valid, torch::Tensor order,
    torch::Tensor cells, torch::Tensor active, torch::Tensor history,
    torch::Tensor hashes, torch::Tensor coefficients, torch::Tensor lengths,
    torch::Tensor rng, int64_t max_cells, int64_t limit, bool native_ties,
    std::vector<torch::Tensor> workspace = {}) {
  for (auto t : {current,goals,candidates,order,cells,hashes,coefficients,lengths,rng})
    check_long_cuda(t,"RSE integer input");
  for (auto t : {goals,candidates,scores,valid,order,cells,active,history,hashes,coefficients,lengths,rng})
    TORCH_CHECK(t.device()==current.device() && t.is_contiguous(),"RSE device/layout mismatch");
  TORCH_CHECK(current.dim()==2 && current.size(0)>0 && current.size(1)>0,"current must be [B,N]");
  auto b=current.size(0), n=current.size(1);
  TORCH_CHECK(goals.sizes()==current.sizes() && order.sizes()==current.sizes(),"goals/order shape");
  TORCH_CHECK(candidates.dim()==3 && candidates.size(0)==b && candidates.size(1)==n && candidates.size(2)==5,"candidates shape");
  TORCH_CHECK(scores.sizes()==candidates.sizes() && scores.scalar_type()==torch::kFloat32,"scores must be float32 [B,N,5]");
  TORCH_CHECK(valid.sizes()==candidates.sizes() && valid.scalar_type()==torch::kBool,"valid shape/type");
  TORCH_CHECK(active.dim()==1 && active.size(0)==b && active.scalar_type()==torch::kBool,"active shape/type");
  TORCH_CHECK(history.dim()==3 && history.size(0)==b && history.size(2)==n && history.scalar_type()==torch::kInt32,"history shape/type");
  auto h=history.size(1);
  TORCH_CHECK(h>0 && hashes.dim()==2 && hashes.size(0)==b && hashes.size(1)==h,"hashes shape");
  TORCH_CHECK(coefficients.dim()==1 && coefficients.size(0)==n,"coefficients shape");
  TORCH_CHECK(cells.dim()==1 && cells.size(0)==b && lengths.dim()==1 && lengths.size(0)==b,"cells/lengths shape");
  TORCH_CHECK(rng.dim()==2 && rng.size(0)==b && rng.size(1)==625,"RNG shape");
  TORCH_CHECK(max_cells>0 && limit>=0,"invalid RSE bounds");
  c10::cuda::CUDAGuard guard(current.device());
  auto opts=current.options();
  // Scratch is caller-owned per shield. Returned tensors remain independently
  // owned, so retaining a previous result across calls is safe.
  if (workspace.empty()) {
    workspace={torch::empty({b,max_cells},opts),torch::empty({b,max_cells},opts),
        torch::empty_like(current),torch::empty_like(current),torch::empty_like(candidates),
        torch::empty_like(valid),torch::empty_like(valid),torch::empty_like(scores),
        torch::empty_like(current),torch::empty_like(current)};
  }
  TORCH_CHECK(workspace.size()==10,"RSE workspace must have 10 tensors");
  for (int i=0; i<10; ++i) {
    auto t=workspace[i];
    auto dtype=(i==5 || i==6)?torch::kBool:i==7?torch::kFloat32:torch::kInt64;
    TORCH_CHECK(t.device()==current.device() && t.is_contiguous() && t.scalar_type()==dtype,
                "RSE workspace device/layout/type mismatch at ",i);
    if (i<2) {
      TORCH_CHECK(t.dim()==2 && t.size(0)==b && t.size(1)==max_cells,"RSE cell workspace shape");
    } else if (i>=4 && i<=7) {
      TORCH_CHECK(t.sizes()==candidates.sizes(),"RSE action workspace shape");
    } else {
      TORCH_CHECK(t.sizes()==current.sizes(),"RSE agent workspace shape");
    }
  }
  auto occ=workspace[0], reserved=workspace[1];
  occ.fill_(-1); reserved.fill_(-1);
  auto next=torch::empty_like(current), actions=torch::empty_like(current);
  auto stack=workspace[2], offsets=workspace[3], preferences=workspace[4];
  auto allowed=workspace[5], forbidden=workspace[6];
  forbidden.zero_();
  auto ties=workspace[7], best_next=workspace[8], best_actions=workspace[9];
  auto stats=torch::zeros({b,4},opts);
  pibt_rse_kernel<<<b,kRSEThreads,0,c10::cuda::getCurrentCUDAStream()>>>(
      current.data_ptr<int64_t>(),goals.data_ptr<int64_t>(),candidates.data_ptr<int64_t>(),
      scores.data_ptr<float>(),valid.data_ptr<bool>(),order.data_ptr<int64_t>(),
      cells.data_ptr<int64_t>(),active.data_ptr<bool>(),history.data_ptr<int32_t>(),
      hashes.data_ptr<int64_t>(),coefficients.data_ptr<int64_t>(),lengths.data_ptr<int64_t>(),
      rng.data_ptr<int64_t>(),occ.data_ptr<int64_t>(),reserved.data_ptr<int64_t>(),
      next.data_ptr<int64_t>(),actions.data_ptr<int64_t>(),stack.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),preferences.data_ptr<int64_t>(),allowed.data_ptr<bool>(),
      forbidden.data_ptr<bool>(),ties.data_ptr<float>(),best_next.data_ptr<int64_t>(),best_actions.data_ptr<int64_t>(),
      stats.data_ptr<int64_t>(),b,n,max_cells,h,limit,native_ties);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {actions,next,stats};
}

std::vector<torch::Tensor> pibt_rse_allocating(
    torch::Tensor current, torch::Tensor goals, torch::Tensor candidates,
    torch::Tensor scores, torch::Tensor valid, torch::Tensor order,
    torch::Tensor cells, torch::Tensor active, torch::Tensor history,
    torch::Tensor hashes, torch::Tensor coefficients, torch::Tensor lengths,
    torch::Tensor rng, int64_t max_cells, int64_t limit, bool native_ties) {
  return pibt_rse(current,goals,candidates,scores,valid,order,cells,active,history,
                  hashes,coefficients,lengths,rng,max_cells,limit,native_ties);
}
