// Native DMM compatibility: reproduce GCC 11 std::sort's observable tie order.
// Independent index-only implementation; priorities are never modified.
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define DMM_ORDER_HD __host__ __device__
#else
#define DMM_ORDER_HD
#endif
namespace dmm_native_order {
DMM_ORDER_HD inline void swap_index(int64_t& a, int64_t& b) {
  int64_t value=a; a=b; b=value;
}
DMM_ORDER_HD inline void sift(int64_t* a, int64_t size, int64_t hole,
                              int64_t value, const float* p) {
  const int64_t top=hole;
  while (2*hole+2<size) {
    int64_t child=2*hole+2;
    if (p[a[child]]>p[a[child-1]]) --child;
    a[hole]=a[child]; hole=child;
  }
  if (2*hole+1<size) { a[hole]=a[2*hole+1]; hole=2*hole+1; }
  while (hole>top && p[a[(hole-1)/2]]>p[value]) {
    a[hole]=a[(hole-1)/2]; hole=(hole-1)/2;
  }
  a[hole]=value;
}
DMM_ORDER_HD inline void heap_sort(int64_t* a, int64_t size, const float* p) {
  if (size<2) return;
  for (int64_t i=(size-2)/2;; --i) {
    sift(a,size,i,a[i],p);
    if (i==0) break;
  }
  for (int64_t end=size-1; end>0; --end) {
    int64_t value=a[end]; a[end]=a[0]; sift(a,end,0,value,p);
  }
}
DMM_ORDER_HD inline void sort(int64_t* a, int64_t n, const float* p) {
  for (int64_t i=0; i<n; ++i) a[i]=i;
  if (n<2) return;
  int depth=0;
  for (int64_t size=n; size>1; size/=2) depth+=2;
  // Right-first introsort; each pending left interval consumes one depth.
  int64_t left_stack[128], right_stack[128]; int depth_stack[128], count=0;
  int64_t left=0, right=n;
  while (true) {
    while (right-left>16) {
      if (depth==0) { heap_sort(a+left,right-left,p); break; }
      --depth;
      int64_t x=left+1, y=left+(right-left)/2, z=right-1, median;
      if (p[a[x]]>p[a[y]])
        median=p[a[y]]>p[a[z]]?y:(p[a[x]]>p[a[z]]?z:x);
      else median=p[a[x]]>p[a[z]]?x:(p[a[y]]>p[a[z]]?z:y);
      swap_index(a[left],a[median]);
      int64_t low=left+1, high=right;
      while (true) {
        while (p[a[low]]>p[a[left]]) ++low;
        do { --high; } while (p[a[left]]>p[a[high]]);
        if (low>=high) break;
        swap_index(a[low++],a[high]);
      }
      left_stack[count]=left; right_stack[count]=low; depth_stack[count++]=depth;
      left=low;
    }
    if (!count) break;
    --count; left=left_stack[count]; right=right_stack[count]; depth=depth_stack[count];
  }
  // Same stable insertion finishing pass, with explicit lower-bound guards.
  for (int64_t i=1; i<n; ++i) {
    int64_t value=a[i], j=i;
    while (j>0 && p[value]>p[a[j-1]]) { a[j]=a[j-1]; --j; }
    a[j]=value;
  }
}
}  // namespace dmm_native_order
#undef DMM_ORDER_HD
