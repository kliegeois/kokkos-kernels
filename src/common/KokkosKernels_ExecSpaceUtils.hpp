/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include "Kokkos_Core.hpp"
#include "Kokkos_Atomic.hpp"

#ifndef _KOKKOSKERNELSUTILSEXECSPACEUTILS_HPP
#define _KOKKOSKERNELSUTILSEXECSPACEUTILS_HPP


namespace KokkosKernels{

namespace Impl{

enum ExecSpaceType{Exec_SERIAL, Exec_OMP, Exec_PTHREADS, Exec_QTHREADS, Exec_CUDA};
template <typename ExecutionSpace>
inline ExecSpaceType kk_get_exec_space_type(){
  ExecSpaceType exec_space = Exec_SERIAL;
#if defined( KOKKOS_ENABLE_SERIAL )
  if (Kokkos::Impl::is_same< Kokkos::Serial , ExecutionSpace >::value){
    exec_space = Exec_SERIAL;
  }
#endif

#if defined( KOKKOS_ENABLE_THREADS )
  if (Kokkos::Impl::is_same< Kokkos::Threads , ExecutionSpace >::value){
    exec_space =  Exec_PTHREADS;
  }
#endif

#if defined( KOKKOS_ENABLE_OPENMP )
  if (Kokkos::Impl::is_same< Kokkos::OpenMP, ExecutionSpace >::value){
    exec_space = Exec_OMP;
  }
#endif

#if defined( KOKKOS_ENABLE_CUDA )
  if (Kokkos::Impl::is_same<Kokkos::Cuda, ExecutionSpace >::value){
    exec_space = Exec_CUDA;
  }
#endif

#if defined( KOKKOS_ENABLE_QTHREAD)
  if (Kokkos::Impl::is_same< Kokkos::Qthread, ExecutionSpace >::value){
    exec_space = Exec_QTHREADS;
  }
#endif
  return exec_space;

}


inline int kk_get_suggested_vector_size(
    const size_t nr, const  size_t nnz, const ExecSpaceType exec_space){
  int suggested_vector_size_ = 1;
  switch (exec_space){
  default:
    break;
  case Exec_SERIAL:
  case Exec_OMP:
  case Exec_PTHREADS:
  case Exec_QTHREADS:
    break;
  case Exec_CUDA:

    if (nr > 0)
      suggested_vector_size_ = nnz / double (nr) + 0.5;
    if (suggested_vector_size_ < 3){
      suggested_vector_size_ = 2;
    }
    else if (suggested_vector_size_ <= 6){
      suggested_vector_size_ = 4;
    }
    else if (suggested_vector_size_ <= 12){
      suggested_vector_size_ = 8;
    }
    else if (suggested_vector_size_ <= 24){
      suggested_vector_size_ = 16;
    }
    else {
      suggested_vector_size_ = 32;
    }
    break;
  }
  return suggested_vector_size_;

}


inline int kk_get_suggested_team_size(const int vector_size, const ExecSpaceType exec_space){
  if (exec_space == Exec_CUDA){
    return 256 / vector_size;
  }
  else {
    return 1;
  }
}

/*
 * CUDA 10.1 graph support. Currently, only supports a linear sequence of kernels.
 * The first definition below is actually a fake wrapper with an identical interface,
 * so that user code for launching kernels only needs to be written once.
 *
 * The second definition is used if the CUDA version supports graphs and the ExecSpace is Cuda.
 *
 * TODO: provide full CUDA graph support, with arbitrary dependencies involing multiple streams, kernels and memcpys.
*/

template<typename ExecSpace, typename LaunchParams>
struct CudaGraphWrapper
{
  CudaGraphWrapper(const CudaGraphWrapper&) = delete;

  //Overload that constructs the params
  template<class... Args>
  bool begin_recording(Args&&... args)
  {
    return begin_recording(LaunchParams(args));
  }

  bool begin_recording(const LaunchParams&)
  {
    //The fake wrapper always re-records, since recording here actually launches the kernels.
    //Even though it's not strictly necessary, keep track of "recording" to catch
    //some simple errors without needing to test with CUDA 10.1.
    if(recording)
      throw std::runtime_error("CudaGraphWrapper: already recording; can't call begin_recording() again");
    return true;
  }

  void end_recording()
  {
    if(!recording)
      throw std::runtime_error("CudaGraphWrapper: not currently recording, so can't call end_recording()");
    recording = false;
  }

  template<typename Tag = void>
  WrappedTeamPolicy team_policy(size_t num_teams, size_t team_size)
  {
    if(!recording)
      throw std::runtime_error("CudaGraphWrapper: Can't create TeamPolicy because begin_recording() has not been called");
    return Kokkos::TeamPolicy<ExecSpace, Tag>(stream, num_teams, team_size);
  }

  template<typename Tag = void>
  WrappedTeamPolicy team_policy(size_t num_teams, size_t team_size, size_t vector_size)
  {
    if(!recording)
      throw std::runtime_error("CudaGraphWrapper: Can't create TeamPolicy because begin_recording() has not been called");
    return Kokkos::TeamPolicy<ExecSpace, Tag>(stream, num_teams, team_size, vector_size);
  }

  template<typename Tag = void>
  WrappedTeamPolicy team_policy(size_t num_teams, size_t team_size, size_t vector_size, size_t sharedPerTeam, size_t sharedPerThread)
  {
    if(!recording)
      throw std::runtime_error("CudaGraphWrapper: Can't create TeamPolicy because begin_recording() has not been called");
    return Kokkos::TeamPolicy<ExecSpace, Tag>(stream, num_teams, team_size).set_scratch_size(0, sharedPerTeam, sharedPerThread);
  }

  template<typename Tag = void>
  WrappedRangePolicy range_policy(size_t begin, size_t end)
  {
    if(!recording)
      throw std::runtime_error("CudaGraphWrapper: Can't create RangePolicy because begin_recording() has not been called");
    return Kokkos::RangePolicy<ExecSpace, Tag>(stream, begin, end);
  }

  void launch() {
    // The kernels were executed during recording, but fence now
    ExecSpace().fence();
  }

  bool recording = false;
};

#if defined(KOKKOS_ENABLE_CUDA) && 10000 < CUDA_VERSION
#define HAVE_CUDAGRAPHS

//Specialize the wrapper for Cuda, to actually use the graph features.
template<typename LaunchParams>
struct CudaGraphWrapper<Kokkos::Cuda, LaunchParams>
{
  //The class assumes these are 
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStream_t stream;

  using P = Kokkos::Experimental::WorkItemProperty::HintLightWeight;
  using WrappedTeamPolicy =
    typename Kokkos::Impl::PolicyPropertyAdaptor<WorkItemProperty::ImplWorkItemProperty<P>,
             Kokkos::TeamPolicy<Kokkos::Cuda>>::policy_out_t>;
  using WrappedRangePolicy =
    typename Kokkos::Impl::PolicyPropertyAdaptor<WorkItemProperty::ImplWorkItemProperty<P>,
             Kokkos::RangePolicy<Kokkos::Cuda>>::policy_out_t>;

  CudaGraphWrapper()
  {
    cudaStreamCreate(&stream);
    graphReady = false;
    recording = false;
  }

  //Don't want to be able to copy this wrapper, because then ownership of the stream gets tricky.
  //Just have one owner and pass around a pointer.
  CudaGraphWrapper(const CudaGraphWrapper&) = delete;

  ~CudaGraphWrapper()
  {
    cudaStreamDestroy(stream);
  }

  //Overload that constructs the params
  template<class... Args>
  bool begin_recording(Args&&... args)
  {
    return begin_recording(LaunchParams(args));
  }

  bool begin_recording(const LaunchParams& params)
  {
    if(graphReady && params == currentParams)
      return false;
    //Otherwise, need to re-record (or record for the first time)
    currentParams = params;
    graphReady = false;
    recording = true;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    return true;
  }

  void end_recording()
  {
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    graphReady = true;
    recording = false;
  }

  template<typename Tag = void>
  WrappedTeamPolicy team_policy(size_t num_teams, size_t team_size)
  {
    if(!recording)
      throw std::runtime_error("CudaGraphWrapper: Can't create TeamPolicy because begin_recording() has not been called");
    return Kokkos::Experimental::require(Kokkos::TeamPolicy<Kokkos::Cuda, Tag>(stream, num_teams, team_size), P);
  }

  template<typename Tag = void>
  WrappedTeamPolicy team_policy(size_t num_teams, size_t team_size, size_t vector_size)
  {
    if(!recording)
      throw std::runtime_error("CudaGraphWrapper: Can't create TeamPolicy because begin_recording() has not been called");
    return Kokkos::Experimental::require(Kokkos::TeamPolicy<Kokkos::Cuda, Tag>(stream, num_teams, team_size, vector_size), P);
  }

  template<typename Tag = void>
  WrappedTeamPolicy team_policy(size_t num_teams, size_t team_size, size_t vector_size, size_t sharedPerTeam, size_t sharedPerThread)
  {
    if(!recording)
      throw std::runtime_error("CudaGraphWrapper: Can't create TeamPolicy because begin_recording() has not been called");
    return Kokkos::Experimental::require(Kokkos::TeamPolicy<Kokkos::Cuda, Tag>(stream, num_teams, team_size).set_scratch_size(0, sharedPerTeam, sharedPerThread), P);
  }

  template<typename Tag = void>
  WrappedRangePolicy range_policy(size_t begin, size_t end)
  {
    if(!recording)
      throw std::runtime_error("CudaGraphWrapper: Can't create RangePolicy because begin_recording() has not been called");
    return Kokkos::Experimental::require(Kokkos::RangePolicy<Kokkos::Cuda, Tag>(stream, begin, end), P);
  }

  //Actually execute all the kernels in the graph
  void launch()
  {
    if(recording)
      throw std::runtime_error("Can't launch CUDA graph while recording kernel launches - call end_recording() first");
    else if(!graphReady)
      throw std::runtime_error("Can't launch CUDA graph: kernels have not been recorded yet.");
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);
  }

  bool recording;
  bool graphReady;
  LaunchParams currentParams;
};

#endif  //HAVE_CUDAGRAPHS

}
}

#endif
