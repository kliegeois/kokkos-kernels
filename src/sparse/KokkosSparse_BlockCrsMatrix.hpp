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

/// \file Kokkos_Sparse_BlockCrsMatrix.hpp
/// \brief Local sparse matrix interface
///
/// This file provides KokkosSparse::BlockCrsMatrix.  This implements a
/// local (no MPI) sparse matrix stored in block compressed row sparse
/// ("Crs") format.

#ifndef KOKKOS_SPARSE_BLOCKCRSMATRIX_HPP_
#define KOKKOS_SPARSE_BLOCKCRSMATRIX_HPP_

#include "Kokkos_Core.hpp"
#include "Kokkos_StaticCrsGraph.hpp"
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include "KokkosSparse_findRelOffset.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

namespace KokkosSparse {

namespace Experimental {

/// \class SparseBlockRowView
/// \brief View of a block-row of a sparse matrix.
/// \tparam MatrixType BlockCrsMatrix Sparse matrix type
///
/// This class provides a generic view of a block-row of a sparse matrix.
///
/// Whether the view is const or not, depends on whether
/// MatrixType is a const or nonconst view of the matrix.  If
/// you always want a const view, use SparseBlockRowViewConst (see below).
///
// TODO ************Add example usage*************
/// Here is an example loop over the entries in the row:
/// \code
/// typedef typename SparseBlockRowView<MatrixType>::value_type value_type;
/// typedef typename SparseBlockRowView<MatrixType>::ordinal_type ordinal_type;
///
/// SparseBlockRowView<MatrixType> A_i = ...;
/// const ordinal_type numEntries = A_i.length;
/// for (ordinal_type k = 0; k < numEntries; ++k) {
///   value_type A_ij = A_i.value (k);
///   ordinal_type j = A_i.colidx (k);
///   // ... do something with A_ij and j ...
/// }
/// \endcode
///
/// MatrixType must provide the \c value_type and \c ordinal_type
/// typedefs.  In addition, it must make sense to use SparseBlockRowView to
/// view a block-row of MatrixType.  
//  TODO *****Update this***** In particular, the values and column
/// indices of a row must be accessible using the <tt>values</tt>
/// resp. <tt>colidx</tt> arrays given to the constructor of this
/// class, with a constant <tt>stride</tt> between successive entries.
template<class MatrixType>
struct SparseBlockRowView {
  //! The type of the values in the row.
  typedef typename MatrixType::value_type value_type;
  //! The type of the column indices in the row.
  typedef typename MatrixType::ordinal_type ordinal_type;
  //! The type for returned block of values. 
  typedef Kokkos::View< value_type**, Kokkos::LayoutStride, typename MatrixType::device_type, Kokkos::MemoryUnmanaged > block_values_type;

private:
  //! Array of values in the row.
  value_type* values_;
  //! Array of (local) column indices in the row.
  ordinal_type* colidx_;
  /// \brief Stride between successive entries in the row.
  ///
  /// For block compressed sparse row (BlockCSR) storage with row-major layout,
  /// (i.e. rows within a block are NOT contiguous)
  /// this is always one. Nevertheless, the stride can never be greater
  /// than the number of rows or columns in the matrix.  Thus,
  /// \c ordinal_type is the correct type.
  // FIXME TODO: Remove stride_ since always set to 1; maybe rename row_in_block_stride to stride...
  const ordinal_type stride_;
  /// \brief Stride between successive rows in a block.
  ///
  /// For block compressed sparse row (BlockCSR) storage with row-major layout by full row,
  /// (i.e. consecutive rows within a block are NOT contiguous), this will be the stride 
  /// between rows within a block-row
  const ordinal_type blockDim_;

public:
  /// \brief Constructor
  ///
  /// \param values [in] Array of the row's values.
  /// \param colidx [in] Array of the row's column indices.
  /// \param stride [in] (Constant) stride between matrix entries in
  ///   each row of a block of the above arrays.
  /// \param blockDim [in] (Constant) stride between block rows
  ///   within a block-row in the above arrays.
  /// \param count [in] Number of blocks in the desired block-row.
  // TODO FIXME: Likely need to remove this constructor - needs additional info not 
  // used here to determine the start of a block-row
  // Assumes offset already at the correct location
  KOKKOS_INLINE_FUNCTION
  SparseBlockRowView (value_type* const values,
                 ordinal_type* const colidx__,
                 const ordinal_type& stride,
                 const ordinal_type& blockDim,
                 const ordinal_type& count) :
    values_ (values), colidx_ (colidx__), stride_ (stride), blockDim_(blockDim), length (count)
  {}

  /// \brief Constructor with offset into \c colidx array
  ///
  /// \param values [in] Array of the row's values.
  /// \param colidx [in] Array of the row's column indices.
  /// \param stride [in] (Constant) stride between matrix entries in
  ///   each row of a block of the above arrays.
  /// \param blockDim [in] (Constant) stride between rows in
  ///   within a block in the above arrays.
  /// \param count [in] Number of blocks in the desired block-row
  /// \param idx [in] Offset into values and colidx of the desired block-row start.
  ///   Note: The offset into the values and colidx arrays for a block-row equals
  ///           num_blocks_prior_to_block-row*blockDim*blockDim
  ///
  /// \tparam OffsetType The type of \c idx (see above).  Must be a
  ///   built-in integer type.  This may differ from ordinal_type.
  ///   For example, the matrix may have dimensions that fit in int,
  ///   but a number of entries that does not fit in int.
  template<class OffsetType>
  KOKKOS_INLINE_FUNCTION
  SparseBlockRowView (const typename MatrixType::values_type& values, // original values view
                 const typename MatrixType::index_type& colidx__, // view from the graph.entries
                 const ordinal_type& stride, // 1
                 const ordinal_type& blockDim,
                 const ordinal_type& count, //num blocks in the row
                 const OffsetType& start,
                 const typename std::enable_if<std::is_integral<OffsetType>::value, int>::type& = 0) :
    values_ (&values(start*blockDim*blockDim)), colidx_ (&colidx__(start)), stride_ (stride), blockDim_(blockDim), length (count)
  {}

  /// \brief Number of entries in the row.
  ///
  /// This is a public const field rather than a public const method,
  /// in order to avoid possible overhead of a method call if the
  /// compiler is unable to inline that method call.
  ///
  /// We assume that rows contain no duplicate entries (i.e., entries
  /// with the same column index).  Thus, a row may have up to
  /// A.numCols() entries.  This means that the correct type of
  /// 'length' is ordinal_type.
  /// Here, length refers to the number of blocks in a block-row
  const ordinal_type length;


  // Return a pointer offset to row i of block K o values_ array; user responsible for indexing into this pointer correctly
  KOKKOS_INLINE_FUNCTION
  value_type* local_row_in_block (const ordinal_type& K, const ordinal_type& i) const {
    //value_type* offset_to_row;
    //return offset_to_row = &(values_[K*blockDim_ + i*length*blockDim_]) ;
    return (values_+(K*blockDim_ + i*length*blockDim_)) ;
  }

  // Return the value for a specified block K with local row,col ids (i,j)
  // Currently, assumes the indices are sorted into blocks (and sorted within the block)
  KOKKOS_INLINE_FUNCTION
  value_type& local_block_value (const ordinal_type& K, const ordinal_type& i, const ordinal_type& j) const {
    //return (local_row_in_block(K,i)[j]); // invalid initializeaiton of non-const reference type
    return values_[K*blockDim_ + i*length*blockDim_ + j];
  }

  // Return unmanaged 2D strided View wrapping block K within this block-row
  KOKKOS_INLINE_FUNCTION
  block_values_type block(const ordinal_type& K) const {
    // Construct with strides
    return block_values_type( &(values_[K*blockDim_]), Kokkos::LayoutStride(blockDim_,length*blockDim_,blockDim_,1) );
  }


  KOKKOS_INLINE_FUNCTION
  ordinal_type findRelBlockOffset ( const ordinal_type idx_to_match, bool is_sorted = false ) {
    ordinal_type offset = -1; // TODO: if ordinal_type is unsigned, what to set it as?
    for ( ordinal_type blk_offset = 0; blk_offset < length; ++blk_offset ) {
      ordinal_type idx = colidx_[blk_offset];
      if ( idx == idx_to_match ) { offset = blk_offset; } // return relative offset
      //if ( idx == idx_to_match ) { offset = blk_offset*blockDim_; } // return offset into values_
    }
  }

#if 0
  /// \brief Reference to the value of entry i in this row of the sparse matrix.
  ///
  /// "Entry i" is not necessarily the entry with column index i, nor
  /// does i necessarily correspond to the (local) row index.
  //
  ///  i in [0, count*blockdim*blockdim)
  ///  offset into values_ and colidx_ already applied for beginning of block-row
  ///  TODO: stride- == 1 - remove
  KOKKOS_INLINE_FUNCTION
  value_type& value (const ordinal_type& i) const {
    return values_[i*stride_];
  }

  /// \brief Reference to the column index of entry i in this row of the sparse matrix.
  ///
  /// "Entry i" is not necessarily the entry with column index i, nor
  /// does i necessarily correspond to the (local) row index.
  KOKKOS_INLINE_FUNCTION
  ordinal_type& colidx (const ordinal_type& i) const {
    return colidx_[i/blockDim_*stride_];
  }
#endif
};


/// \class SparseBlockRowViewConst
/// \brief Const view of a row of a sparse matrix.
/// \tparam MatrixType Sparse matrix type, such as (but not limited to) BlockCrsMatrix.
///
/// This class is like SparseBlockRowView, except that it provides a const
/// view.  This class exists in order to let users get a const view of
/// a row of a nonconst matrix.
template<class MatrixType>
struct SparseBlockRowViewConst {
  //! The type of the values in the row.
  typedef const typename MatrixType::non_const_value_type value_type;
  //! The type of the column indices in the row.
  typedef const typename MatrixType::non_const_ordinal_type ordinal_type;
  //! The type for returned block of values. 
  typedef Kokkos::View< value_type**, Kokkos::LayoutStride, typename MatrixType::device_type, Kokkos::MemoryUnmanaged > block_values_type;

private:
  //! Array of values in the row.
  value_type* values_;
  //! Array of (local) column indices in the row.
  ordinal_type* colidx_;
  /// \brief Stride between successive entries in the row.
  ///
  /// For compressed sparse row (CSR) storage, this is always one.
  /// This might be greater than one for storage formats like ELLPACK
  /// or Jagged Diagonal.  Nevertheless, the stride can never be
  /// greater than the number of rows or columns in the matrix.  Thus,
  /// \c ordinal_type is the correct type.
  const ordinal_type stride_;
  /// \brief Stride between successive rows in a block-row
  ///
  /// For block compressed sparse row (BlockCSR) storage with row-major layout,
  /// (i.e. consecutive rows within a block are NOT contiguous), this will be the stride 
  /// between rows within a block-row
  const ordinal_type blockDim_;

public:
  /// \brief Constructor
  ///
  /// \param values [in] Array of the row's values.
  /// \param colidx [in] Array of the row's column indices.
  /// \param stride [in] (Constant) stride between matrix entries in
  ///   each of the above arrays.
  /// \param count [in] Number of entries in the row.
  KOKKOS_INLINE_FUNCTION
  SparseBlockRowViewConst (value_type* const values,
                      ordinal_type* const colidx__,
                      const ordinal_type& stride,
                      const ordinal_type& blockDim,
                      const ordinal_type& count) :
    values_ (values), colidx_ (colidx__), stride_ (stride), blockDim_(blockDim), length (count)
  {}

  /// \brief Constructor with offset into \c colidx array
  ///
  /// \param values [in] Array of the row's values.
  /// \param colidx [in] Array of the row's column indices.
  /// \param stride [in] (Constant) stride between matrix entries in
  ///   each of the above arrays.
  /// \param count [in] Number of entries in the row.
  /// \param idx [in] Start offset into \c colidx array
  ///
  /// \tparam OffsetType The type of \c idx (see above).  Must be a
  ///   built-in integer type.  This may differ from ordinal_type.
  ///   For example, the matrix may have dimensions that fit in int,
  ///   but a number of entries that does not fit in int.
  template<class OffsetType>
  KOKKOS_INLINE_FUNCTION
  SparseBlockRowViewConst (const typename MatrixType::values_type& values,
                      const typename MatrixType::index_type& colidx__,
                      const ordinal_type& stride,
                      const ordinal_type& blockDim,
                      const ordinal_type& count,
                      const OffsetType& start,
                      const typename std::enable_if<std::is_integral<OffsetType>::value, int>::type& = 0) :
    values_ (&values(start*blockDim*blockDim)), colidx_ (&colidx__(start)), stride_ (stride), blockDim_(blockDim), length (count)
  {}

  /// \brief Number of entries in the row.
  ///
  /// This is a public const field rather than a public const method,
  /// in order to avoid possible overhead of a method call if the
  /// compiler is unable to inline that method call.
  ///
  /// We assume that rows contain no duplicate entries (i.e., entries
  /// with the same column index).  Thus, a row may have up to
  /// A.numCols() entries.  This means that the correct type of
  /// 'length' is ordinal_type.
  const ordinal_type length;


  // Return a pointer offset to row i of block K o values_ array; user responsible for indexing into this pointer correctly
  KOKKOS_INLINE_FUNCTION
  value_type* local_row_in_block (const ordinal_type& K, const ordinal_type& i) const {
    //value_type* offset_to_row;
    //return offset_to_row = &(values_[K*blockDim_ + i*length*blockDim_]) ;
    return (values_+(K*blockDim_ + i*length*blockDim_)) ;
  }

  // Return the value for a specified block K with local row,col ids (i,j)
  // Currently, assumes the indices are sorted into blocks (and sorted within the block)
  KOKKOS_INLINE_FUNCTION
  value_type& local_block_value (const ordinal_type& K, const ordinal_type& i, const ordinal_type& j) const {
    //return (local_row_in_block(K,i)[j]);
    return values_[K*blockDim_ + i*length*blockDim_ + j];
  }

  // Return unmanaged 2D strided View wrapping block K within this block-row
  KOKKOS_INLINE_FUNCTION
  block_values_type block(const ordinal_type& K) const {
    // Construct stride
    return block_values_type( &(values_[K*blockDim_]), Kokkos::LayoutStride(blockDim_,length*blockDim_,blockDim_,1) );
  }


  KOKKOS_INLINE_FUNCTION
  ordinal_type findRelBlockOffset ( const ordinal_type idx_to_match, bool is_sorted = false ) {
    ordinal_type offset = -1; // TODO: if ordinal_type is unsigned, what to set it as?
    for ( ordinal_type blk_offset = 0; blk_offset < length; ++blk_offset ) {
      ordinal_type idx = colidx_[blk_offset];
      if ( idx == idx_to_match ) { offset = blk_offset; } // return relative offset
      //if ( idx == idx_to_match ) { offset = blk_offset*blockDim_; } // return offset into values_
    }
  }

#if 0
  /// \brief (Const) reference to the value of entry i in this row of
  ///   the sparse matrix.
  ///
  /// "Entry i" is not necessarily the entry with column index i, nor
  /// does i necessarily correspond to the (local) row index.
  //
  ///  i in [0, count*blockdim*blockdim)
  ///  offset into values_ and colidx_ already applied for beginning of block-row
  ///  TODO: stride- == 1 - remove
  KOKKOS_INLINE_FUNCTION
  value_type& value (const ordinal_type& i) const {
    return values_[i*stride_];
  }

  /// \brief (Const) reference to the column index of entry i in this
  ///   row of the sparse matrix.
  ///
  /// "Entry i" is not necessarily the entry with column index i, nor
  /// does i necessarily correspond to the (local) row index.
  KOKKOS_INLINE_FUNCTION
  ordinal_type& colidx (const ordinal_type& i) const {
    return colidx_[i/blockDim_*stride_];
  }
#endif
};

/// \class BlockCrsMatrix
/// \brief Compressed sparse row implementation of a sparse matrix.
/// \tparam ScalarType The type of entries in the sparse matrix.
/// \tparam OrdinalType The type of column indices in the sparse matrix.
/// \tparam Device The Kokkos Device type.
/// \tparam MemoryTraits Traits describing how Kokkos manages and
///   accesses data.  The default parameter suffices for most users.
///
/// "Crs" stands for "compressed row sparse."  This is the phrase
/// Trilinos traditionally uses to describe compressed sparse row
/// storage for sparse matrices, as described, for example, in Saad
/// (2nd ed.).
template<class ScalarType,
         class OrdinalType,
         class Device,
         class MemoryTraits = void,
         class SizeType = typename Kokkos::ViewTraits<OrdinalType*, Device, void, void>::size_type>
class BlockCrsMatrix {
private:
  typedef typename Kokkos::ViewTraits<ScalarType*,Device,void,void>::host_mirror_space host_mirror_space ;
public:
  //! Type of the matrix's execution space.
  typedef typename Device::execution_space execution_space;
  //! Type of the matrix's memory space.
  typedef typename Device::memory_space memory_space;
  //! Type of the matrix's device type.
  typedef Kokkos::Device<execution_space, memory_space> device_type;

  //! Type of each value in the matrix.
  typedef ScalarType value_type;
  //! Type of each (column) index in the matrix.
  typedef OrdinalType ordinal_type;
  typedef MemoryTraits memory_traits;
  /// \brief Type of each entry of the "row map."
  ///
  /// The "row map" corresponds to the \c ptr array of row offsets in
  /// compressed sparse row (CSR) storage.
  typedef SizeType size_type;

  //! Type of a host-memory mirror of the sparse matrix.
  typedef BlockCrsMatrix<ScalarType, OrdinalType, host_mirror_space, MemoryTraits> HostMirror;
  //! Type of the graph structure of the sparse matrix.
  typedef Kokkos::StaticCrsGraph<OrdinalType, Kokkos::LayoutLeft, execution_space, SizeType> StaticCrsGraphType;
  //! Type of column indices in the sparse matrix.
  typedef typename StaticCrsGraphType::entries_type index_type;
  //! Const version of the type of column indices in the sparse matrix.
  typedef typename index_type::const_value_type const_ordinal_type;
  //! Nonconst version of the type of column indices in the sparse matrix.
  typedef typename index_type::non_const_value_type non_const_ordinal_type;
  //! Type of the "row map" (which contains the offset for each row's data).
  typedef typename StaticCrsGraphType::row_map_type row_map_type;
  //! Const version of the type of row offsets in the sparse matrix.
  typedef typename row_map_type::const_value_type const_size_type;
  //! Nonconst version of the type of row offsets in the sparse matrix.
  typedef typename row_map_type::non_const_value_type non_const_size_type;
  //! Kokkos Array type of the entries (values) in the sparse matrix.
  typedef Kokkos::View<value_type*, Kokkos::LayoutRight, device_type, MemoryTraits> values_type;
  //! Const version of the type of the entries in the sparse matrix.
  typedef typename values_type::const_value_type const_value_type;
  //! Nonconst version of the type of the entries in the sparse matrix.
  typedef typename values_type::non_const_value_type non_const_value_type;

#ifdef KOKKOS_USE_CUSPARSE
  cusparseHandle_t cusparse_handle;
  cusparseMatDescr_t cusparse_descr;
#endif // KOKKOS_USE_CUSPARSE

  /// \name Storage of the actual sparsity structure and values.
  ///
  /// BlockCrsMatrix uses the compressed sparse row (CSR) storage format to
  /// store the sparse matrix.  CSR is also called "compressed row
  /// storage"; hence the name, which it inherits from Tpetra and from
  /// Epetra before it.
  //@{
  //! The graph (sparsity structure) of the sparse matrix.
  StaticCrsGraphType graph;
  //! The 1-D array of values of the sparse matrix.
  values_type values;
  //@}

  /// \brief Launch configuration that can be used by
  ///   overloads/specializations of MV_multiply().
  ///
  /// This is a hack and needs to be replaced by a general
  /// state mechanism.
  DeviceConfig dev_config;

  /// \brief Default constructor; constructs an empty sparse matrix.
  ///
  /// FIXME (mfh 09 Aug 2013) numCols and nnz should be properties of
  /// the graph, not the matrix.  Then BlockCrsMatrix needs methods to get
  /// these from the graph.
  BlockCrsMatrix () :
    numCols_ (0),
    blockDim_ (0)
  {}

  //! Copy constructor (shallow copy).
  template<typename SType,
           typename OType,
           class DType,
           class MTType,
           typename IType>
  BlockCrsMatrix (const BlockCrsMatrix<SType,OType,DType,MTType,IType> & B) :
    graph (B.graph.entries, B.graph.row_map),
    values (B.values),
    dev_config (B.dev_config),
#ifdef KOKKOS_USE_CUSPARSE
    cusparse_handle (B.cusparse_handle),
    cusparse_descr (B.cusparse_descr),
#endif // KOKKOS_USE_CUSPARSE
    numCols_ (B.numCols ()),
    blockDim_ (B.blockDim ())
  {
    graph.row_block_offsets = B.graph.row_block_offsets;
    //TODO: MD 07/2017: Changed the copy constructor of graph
    //as the constructor of StaticCrsGraph does not allow copy from non const version.
  }

  /// \brief Construct with a graph that will be shared.
  ///
  /// Allocate the values array for subsquent fill.
  // TODO: Require graph must be for the block matrix, not CRS graph...
  BlockCrsMatrix (const std::string& arg_label,
             const StaticCrsGraphType& arg_graph, const OrdinalType& blockDim) :
    graph (arg_graph),
    values (arg_label, arg_graph.entries.extent(0)),
    numCols_ (maximum_entry (arg_graph) + 1),
    blockDim_ (blockDim)
  {}

  /// \brief Constructor that copies raw arrays of host data in
  ///   coordinate format.
  ///
  /// On input, each entry of the sparse matrix is stored in val[k],
  /// with row index rows[k] and column index cols[k].  We assume that
  /// the entries are sorted in increasing order by row index.
  ///
  /// This constructor is mainly useful for benchmarking or for
  /// reading the sparse matrix's data from a file.
  ///
  /// \param label [in] The sparse matrix's label.
  /// \param nrows [in] The number of rows.
  /// \param ncols [in] The number of columns.
  /// \param annz [in] The number of entries.
  /// \param val [in] The entries.
  /// \param rows [in] The row indices.  rows[k] is the row index of
  ///   val[k].
  /// \param cols [in] The column indices.  cols[k] is the column
  ///   index of val[k].
  /// \param pad [in] If true, pad the sparse matrix's storage with
  ///   zeros in order to improve cache alignment and / or
  ///   vectorization.
  ///
  /// FIXME (mfh 21 Jun 2013) The \c pad argument is currently not used.
  BlockCrsMatrix (const std::string &label,
             OrdinalType nrows,
             OrdinalType ncols,
             size_type annz,
             ScalarType* val,
             OrdinalType* rows,
             OrdinalType* cols,
             OrdinalType blockdim,
             bool pad = false)
  {
    (void) pad;
    import (label, nrows, ncols, annz, val, rows, cols, blockdim);

    // FIXME (mfh 09 Aug 2013) Specialize this on the Device type.
    // Only use cuSPARSE for the Cuda Device.
#ifdef KOKKOS_USE_CUSPARSE
    // FIXME (mfh 09 Aug 2013) This is actually static initialization
    // of the library; you should do it once for the whole program,
    // not once per matrix.  We need to protect this somehow.
    cusparseCreate (&cusparse_handle);

    // This is a per-matrix attribute.  It encapsulates things like
    // whether the matrix is lower or upper triangular, etc.  Ditto
    // for other TPLs like MKL.
    cusparseCreateMatDescr (&cusparse_descr);
#endif // KOKKOS_USE_CUSPARSE
  }

  /// \brief Constructor that accepts a row map, column indices, and
  ///   values.
  ///
  /// The matrix will store and use the row map, indices, and values
  /// directly (by view, not by deep copy).
  ///
  /// \param label [in] The sparse matrix's label.
  /// \param nrows [in] The number of rows.
  /// \param ncols [in] The number of columns.
  /// \param annz [in] The number of entries.
  /// \param vals [in/out] The entries.
  /// \param rows [in/out] The row map (containing the offsets to the
  ///   data in each row).
  /// \param cols [in/out] The column indices.

  BlockCrsMatrix (const std::string& label,
             const OrdinalType nrows,
             const OrdinalType ncols,
             const size_type annz,
             const values_type& vals,
             const row_map_type& rows,
             const index_type& cols,
             const OrdinalType blockDim) :
    graph (cols, rows),
    values (vals),
    numCols_ (ncols),
    blockDim_ (blockDim)
  {

    const ordinal_type actualNumRows = (rows.extent (0) != 0) ?
      static_cast<ordinal_type> (rows.extent (0) - static_cast<size_type> (1)) :
      static_cast<ordinal_type> (0);
    if (nrows != actualNumRows) {
      std::ostringstream os;
      os << "Input argument nrows = " << nrows << " != the actual number of "
        "rows " << actualNumRows << " according to the 'rows' input argument.";
      throw std::invalid_argument (os.str ());
    }
    // nnz returns graph.entries.extent(0) i.e. ptr[ nrows + 1 ] nnz entry
    // input annz is nnz of values, not comparable with block ptr 'nnz' i.e. numBlocks
    /*
    if (annz != nnz ()) {
      std::ostringstream os;
      os << "Input argument annz = " << annz
         << " != this->nnz () = " << nnz () << ".";
      throw std::invalid_argument (os.str ());
    }
    */
    if (blockDim_ <= 0) {
      std::ostringstream os;
      os << "Input argument blockDim = " << blockDim 
         << " is not larger than 0.";
      throw std::invalid_argument (os.str ());
    }

#ifdef KOKKOS_USE_CUSPARSE
    cusparseCreate (&cusparse_handle);
    cusparseCreateMatDescr (&cusparse_descr);
#endif // KOKKOS_USE_CUSPARSE
  }

  /// \brief Constructor that accepts a a static graph, and values.
  ///
  /// The matrix will store and use the row map, indices, and values
  /// directly (by view, not by deep copy).
  ///
  /// \param label [in] The sparse matrix's label.
  /// \param nrows [in] The number of rows.
  /// \param ncols [in] The number of columns.
  /// \param annz [in] The number of entries.
  /// \param vals [in/out] The entries.
  /// \param rows [in/out] The row map (containing the offsets to the
  ///   data in each row).
  /// \param cols [in/out] The column indices.

  BlockCrsMatrix (const std::string& label,
             const OrdinalType& ncols,
             const values_type& vals,
             const StaticCrsGraphType& graph_,
             const OrdinalType& blockDim) :
    graph (graph_),
    values (vals),
    numCols_ (ncols),
    blockDim_ (blockDim)
  {
#ifdef KOKKOS_USE_CUSPARSE
    cusparseCreate (&cusparse_handle);
    cusparseCreateMatDescr (&cusparse_descr);
#endif // KOKKOS_USE_CUSPARSE
  }

  void
  import (const std::string &label,
          const OrdinalType nrows,
          const OrdinalType ncols,
          const size_type annz,
          ScalarType* val,
          OrdinalType* rows,
          OrdinalType* cols,
          const OrdinalType blockDim);

  // Input:
  // rowi   is a block-row index
  // ncol   is number of blocks referenced in cols[] array
  // cols[] are block colidxs within the block-row to be summed into
  //        ncol entries
  // vals[] array containing 'block' of values
  //        ncol*block_size*block_size entries
  //        assume vals block is provided in 'LayoutRight' or 'Row Major' format, that is 
  //        e.g. 2x2 block [ a b ; c d ] provided as flattened 1d array as [a b c d]
  //        TODO: Confirm that each block is stored contiguously in vals:
  //        [a b; c d] [e f; g h] -> [a b c d e f g h]
  //        If so, then i in [0, ncols) for cols[] 
  //        maps to i*block_size*block_size in vals[]
  KOKKOS_INLINE_FUNCTION
  OrdinalType
  sumIntoValues (const OrdinalType rowi,
                 const OrdinalType cols[],
                 const OrdinalType ncol,
                 const ScalarType vals[],
                 const bool is_sorted = false,
                 const bool force_atomic = false) const
  {
    SparseBlockRowView<BlockCrsMatrix> row_view = this->block_row (rowi);
    const ordinal_type length = row_view.length; // num blocks in block-row rowi
    const ordinal_type block_size = this->blockDim();

    //ordinal_type hint = 0; // Guess for offset of current column index in row
    ordinal_type numValid = 0; // number of valid local column indices

    for (ordinal_type i = 0; i < ncol; ++i) {

      // Find offset into values for block-row rowi and colidx cols[i]
      // cols[i] is the index to match
      // blk_offset is the offset for block colidx from bptr[rowi] to bptr[rowi + 1] (not global offset)
      // colidx_ and values_ are already offset to the beginning of blockrow rowi
      auto blk_offset = findRelBlockOffset(cols[i], is_sorted);
      ordinal_type offset_into_values = blk_offset*block_size;
      if ( offset_into_values != -1 ) {
        ordinal_type offset_into_vals = i*block_size*block_size; //stride == 1 assumed between elements
        ordinal_type values_row_stride = block_size*length; // stride to start of next row
        for ( ordinal_type lrow = 0; lrow < block_size; ++lrow ) {
          auto local_row_values = row_view.local_row_in_block(blk_offset, lrow); // pointer to start of specified local row within this block
          for ( ordinal_type lcol = 0; lcol < block_size; ++lcol ) {
            if (force_atomic) {
              Kokkos::atomic_add (&(local_row_values[lcol]), vals[ offset_into_vals + lrow*block_size + lcol ]);
            }
            else {
              local_row_values[lcol] += vals[ offset_into_vals + lrow*block_size + lcol];
              //values_[ offset_into_values + lrow*values_row_stride + lcol] += vals[ offset_into_vals + lrow*block_size + lcol];
            }
          }
        }
        ++numValid;
      }

#if 0
      // FIXME pseudo-code thought-gathering details - remove
      for ( ordinal_type blk_offset = 0; blk_offset < length; ++blk_offset ) {
        ordinal_type idx = row_view.colidx_[blk_offset];
        if ( idx == cols[i] ) {
          //found, matched
          ordinal_type offset_into_vals = i*block_size*block_size; //stride == 1 assumed between elements
          ordinal_type offset_into_values = blk_offset*block_size;
          ordinal_type values_row_stride = block_size*length; // stride to start of next row

          for ( ordinal_type lrow = 0; lrow < block_size; ++lrow ) {
            auto local_row_values = row_view.local_row_in_block(blk_offset, lrow);
            for ( ordinal_type lcol = 0; lcol < block_size; ++lcol ) {
              local_row_values[lcol] += vals[ offset_into_vals + lrow*block_size + lcol];
              //values_[ offset_into_values + lrow*values_row_stride + lcol] += vals[ offset_into_vals + lrow*block_size + lcol];
            }
          }
        }
      }

      // FIXME Previous implementation - remove
      // NOTE (mfh 19 Sep 2017) This assumes that row_view stores
      // column indices contiguously.  It does, but one could imagine
      // changing that at some point.
      const ordinal_type offset =
        findRelOffset (&(row_view.colidx(0)), length, cols[i], hint, is_sorted);
      if (offset != length) {
        if (force_atomic) {
          Kokkos::atomic_add (&(row_view.value(offset)), vals[i]);
        }
        else {
          row_view.value(offset) += vals[i];
        }
        ++numValid;
        // If the hint is out of range, findRelOffset will ignore it.
        // Thus, while it's harmless to have a hint out of range, it
        // may slow down searches for subsequent valid input column
        // indices.
        hint = offset + 1;
      }
#endif

    } // end for ncol
    return numValid;
  }


#if 1
  KOKKOS_INLINE_FUNCTION
  OrdinalType
  replaceValues (const OrdinalType rowi,
                 const OrdinalType cols[],
                 const OrdinalType ncol,
                 const ScalarType vals[],
                 const bool is_sorted = false,
                 const bool force_atomic = false) const
  {
    SparseBlockRowView<BlockCrsMatrix> row_view = this->row (rowi);
    const ordinal_type length = row_view.length;

    //ordinal_type hint = 0; // Guess for offset of current column index in row
    ordinal_type numValid = 0; // number of valid local column indices

    for (ordinal_type i = 0; i < ncol; ++i) {

      // Find offset into values for block-row rowi and colidx cols[i]
      // cols[i] is the index to match
      // blk_offset is the offset for block colidx from bptr[rowi] to bptr[rowi + 1] (not global offset)
      // colidx_ and values_ are already offset to the beginning of blockrow rowi
      auto blk_offset = findRelBlockOffset(cols[i], is_sorted);
      ordinal_type offset_into_values = blk_offset*block_size;
      if ( offset_into_values != -1 ) {
        ordinal_type offset_into_vals = i*block_size*block_size; //stride == 1 assumed between elements
        ordinal_type values_row_stride = block_size*length; // stride to start of next row
        for ( ordinal_type lrow = 0; lrow < block_size; ++lrow ) {
          auto local_row_values = row_view.local_row_in_block(blk_offset, lrow); // pointer to start of specified local row within this block
          for ( ordinal_type lcol = 0; lcol < block_size; ++lcol ) {
            if (force_atomic) {
              Kokkos::atomic_assign(&(local_row_values[lcol]), vals[ offset_into_vals + lrow*block_size + lcol ]);
            }
            else {
              local_row_values[lcol] = vals[ offset_into_vals + lrow*block_size + lcol];
              //values_[ offset_into_values + lrow*values_row_stride + lcol] = vals[ offset_into_vals + lrow*block_size + lcol];
            }
          }
        }

#if 0
      // NOTE (mfh 19 Sep 2017) This assumes that row_view stores
      // column indices contiguously.  It does, but one could imagine
      // changing that at some point.
      const ordinal_type offset =
        findRelOffset (&(row_view.colidx(0)), length, cols[i], hint, is_sorted);
      if (offset != length) {
        if (force_atomic) {
          Kokkos::atomic_assign (&(row_view.value(offset)), vals[i]);
        }
        else {
          row_view.value(offset) = vals[i];
        }
        ++numValid;
        // If the hint is out of range, findRelOffset will ignore it.
        // Thus, while it's harmless to have a hint out of range, it
        // may slow down searches for subsequent valid input column
        // indices.
        hint = offset + 1;
      }
#endif

    } // end for ncol
    return numValid;
  }
#endif

  //! Attempt to assign the input matrix to \c *this.
  template<typename aScalarType, typename aOrdinalType, class aDevice, class aMemoryTraits,typename aSizeType>
  BlockCrsMatrix&
  operator= (const BlockCrsMatrix<aScalarType, aOrdinalType, aDevice, aMemoryTraits, aSizeType>& mtx)
  {
    numCols_ = mtx.numCols ();
    blockDim_ = mtx.blockDim ();
    graph = mtx.graph;
    values = mtx.values;
    dev_config = mtx.dev_config;
    return *this;
  }

  //! The number of rows in the sparse matrix.
  KOKKOS_INLINE_FUNCTION ordinal_type numRows () const {
    return graph.numRows ();
  }

  //! The number of columns in the sparse matrix.
  KOKKOS_INLINE_FUNCTION ordinal_type numCols () const {
    return numCols_;
  }

  //! The block dimension in the sparse block matrix.
  KOKKOS_INLINE_FUNCTION ordinal_type blockDim () const {
    return blockDim_ ;
  }

  //! The number of stored entries in the sparse matrix.
  KOKKOS_INLINE_FUNCTION size_type nnz () const {
    return graph.entries.extent (0);
  }

  friend struct SparseBlockRowView<BlockCrsMatrix>;
  friend struct KokkosSparse::SparseRowView<BlockCrsMatrix>;

  /// \brief Return a view of row i of the matrix.
  ///
  /// If row i does not belong to the matrix, return an empty view.
  ///
  /// The returned object \c view implements the following interface:
  /// <ul>
  /// <li> \c view.length is the number of entries in the row </li>
  /// <li> \c view.value(k) returns a nonconst reference
  ///      to the value of the k-th entry in the row </li>
  /// <li> \c view.colidx(k) returns a nonconst reference to
  ///      the column index of the k-th entry in the row </li>
  /// </ul>
  /// k is not a column index; it just counts from 0 to
  /// <tt>view.length - 1</tt>.
  ///
  /// Users should not rely on the return type of this method.  They
  /// should instead assign to 'auto'.
  ///
  /// Both row() and rowConst() used to take a "SizeType" template
  /// parameter, which was the type to use for row offsets.  This is
  /// unnecessary, because the BlockCrsMatrix specialization already has
  /// the row offset type available, via the <tt>size_type</tt>
  /// typedef.  Our sparse matrix-vector multiply implementation for
  /// BlockCrsMatrix safely uses <tt>ordinal_type</tt> rather than
  /// <tt>size_type</tt> to iterate over all the entries in a row of
  /// the sparse matrix.  Since <tt>ordinal_type</tt> may be smaller
  /// than <tt>size_type</tt>, compilers may generate more efficient
  /// code.  The row() and rowConst() methods first compute the
  /// difference of consecutive row offsets as <tt>size_type</tt>, and
  /// then cast to <tt>ordinal_type</tt>.  If you want to do this
  /// yourself, here is an example:
  ///
  /// \code
  /// for (ordinal_type lclRow = 0; lclRow < A.numRows (); ++lclRow) {
  ///   const ordinal_type numEnt =
  ///     static_cast<ordinal_type> (A.graph.row_map(i+1) - A.graph.row_map(i));
  ///   for (ordinal_type k = 0; k < numEnt; ++k) {
  ///     // etc.
  ///   }
  /// }
  /// \endcode
  KOKKOS_INLINE_FUNCTION
  SparseBlockRowView<BlockCrsMatrix> row (const ordinal_type i) const {
    const size_type start = graph.row_map(i);
    // count is guaranteed to fit in ordinal_type, as long as no row
    // has duplicate entries.
    const ordinal_type count = static_cast<ordinal_type> (graph.row_map(i+1) - start);

    // TODO: Should this still be compatible with SparseRowView from CrsMatrix?
    if (count == 0) {
      return SparseRowView<BlockCrsMatrix> (NULL, NULL, 1, 0);
    } else {
      return SparseRowView<BlockCrsMatrix> (values, graph.entries, 1, count, start);
    }
  }

  KOKKOS_INLINE_FUNCTION
  SparseBlockRowView<BlockCrsMatrix> block_row (const ordinal_type i) const {
    const size_type start = graph.row_map(i); // total num blocks to this point
    const size_type start_offset = graph.row_map(i)*blockDim()*blockDim(); // total num blocks to this point * blockDim^2 for offset into colidx and values arrays

    // count is guaranteed to fit in ordinal_type, as long as no row
    // has duplicate entries.
    const ordinal_type count = static_cast<ordinal_type> (graph.row_map(i+1) - start); // num blocks in this row
    const ordinal_type row_strides_in_block = static_cast<ordinal_type> ( blockDim() ); // stride between rows in a block-row

    std::cout << "br i = " << i << "  start = " << start << "  count = " << count << "  row_strides_in_block = " << row_strides_in_block << "  start_offset = " << start_offset<< std::endl;
    if (count == 0) {
      return SparseBlockRowView<BlockCrsMatrix> (NULL, NULL, 1, 1, 0);
    } else {
      //return SparseBlockRowView<BlockCrsMatrix> (values, graph.entries, 1, row_strides_in_block, count, start, start_offset);
      return SparseBlockRowView<BlockCrsMatrix> (values, graph.entries, 1, row_strides_in_block, count, start);
    }
  }

  /// \brief Return a const view of row i of the matrix.
  ///
  /// If row i does not belong to the matrix, return an empty view.
  ///
  /// The returned object \c view implements the following interface:
  /// <ul>
  /// <li> \c view.length is the number of entries in the row </li>
  /// <li> \c view.value(k) returns a const reference to
  ///      the value of the k-th entry in the row </li>
  /// <li> \c view.colidx(k) returns a const reference to the
  ///      column index of the k-th entry in the row </li>
  /// </ul>
  /// k is not a column index; it just counts from 0 to
  /// <tt>view.length - 1</tt>.
  ///
  /// Users should not rely on the return type of this method.  They
  /// should instead assign to 'auto'.
  ///
  /// Both row() and rowConst() used to take a "SizeType" template
  /// parameter, which was the type to use for row offsets.  This is
  /// unnecessary, because the BlockCrsMatrix specialization already has
  /// the row offset type available, via the <tt>size_type</tt>
  /// typedef.  Our sparse matrix-vector multiply implementation for
  /// BlockCrsMatrix safely uses <tt>ordinal_type</tt> rather than
  /// <tt>size_type</tt> to iterate over all the entries in a row of
  /// the sparse matrix.  Since <tt>ordinal_type</tt> may be smaller
  /// than <tt>size_type</tt>, compilers may generate more efficient
  /// code.  The row() and rowConst() methods first compute the
  /// difference of consecutive row offsets as <tt>size_type</tt>, and
  /// then cast to <tt>ordinal_type</tt>.  If you want to do this
  /// yourself, here is an example:
  ///
  /// \code
  /// for (ordinal_type lclRow = 0; lclRow < A.numRows (); ++lclRow) {
  ///   const ordinal_type numEnt =
  ///     static_cast<ordinal_type> (A.graph.row_map(i+1) - A.graph.row_map(i));
  ///   for (ordinal_type k = 0; k < numEnt; ++k) {
  ///     // etc.
  ///   }
  /// }
  /// \endcode
  KOKKOS_INLINE_FUNCTION
  SparseBlockRowViewConst<BlockCrsMatrix> rowConst (const ordinal_type i) const {
    const size_type start = graph.row_map(i);

    // count is guaranteed to fit in ordinal_type, as long as no row
    // has duplicate entries.
    const ordinal_type count = static_cast<ordinal_type> (graph.row_map(i+1) - start);

    if (count == 0) {
      return SparseRowViewConst<BlockCrsMatrix> (NULL, NULL, 1, 0);
    } else {
      return SparseRowViewConst<BlockCrsMatrix> (values, graph.entries, 1, count, start);
    }
  }

  KOKKOS_INLINE_FUNCTION
  SparseBlockRowViewConst<BlockCrsMatrix> block_row_Const (const ordinal_type i) const {
    const size_type start = graph.row_map(i); // total num blocks to this point
    const size_type start_offset = graph.row_map(i)*blockDim()*blockDim(); // total num blocks to this point * blockDim^2 for offset into colidx and values arrays
    const ordinal_type count = static_cast<ordinal_type> (graph.row_map(i+1) - start); // num blocks in this row
    const ordinal_type row_strides_in_block = static_cast<ordinal_type> ( blockDim() ); // stride between rows in a block-row

    //std::cout << "br i = " << i << "  start = " << start << "  count = " << count << "  row_strides_in_block = " << row_strides_in_block << "  start_offset = " << start_offset<< std::endl;
    if (count == 0) {
      return SparseBlockRowViewConst<BlockCrsMatrix> (NULL, NULL, 1, 1, 0);
    } else {
      //return SparseBlockRowViewConst<BlockCrsMatrix> (values, graph.entries, 1, row_strides_in_block, count, start, start_offset);
      return SparseBlockRowViewConst<BlockCrsMatrix> (values, graph.entries, 1, row_strides_in_block, count, start);
    }
  }

private:
  ordinal_type numCols_;
  ordinal_type blockDim_; // TODO Assuming square blocks for now...
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template< typename ScalarType , typename OrdinalType, class Device, class MemoryTraits, typename SizeType >
void
BlockCrsMatrix<ScalarType , OrdinalType, Device, MemoryTraits, SizeType >::
import (const std::string &label,
        const OrdinalType nrows,
        const OrdinalType ncols,
        const size_type annz,
        ScalarType* val,
        OrdinalType* rows,
        OrdinalType* cols,
        const OrdinalType blockDim)
{
  std::string str = label;
  values = values_type (str.append (".values"), annz);

  numCols_ = ncols;
  blockDim_ = blockDim;

  // FIXME (09 Aug 2013) CrsArray only takes std::vector for now.
  // We'll need to fix that.
  std::vector<int> row_lengths (nrows, 0);

  // FIXME (mfh 21 Jun 2013) This calls for a parallel_for kernel.
  for (OrdinalType i = 0; i < nrows; ++i) {
    row_lengths[i] = rows[i + 1] - rows[i];
  }

  str = label;
  graph = Kokkos::create_staticcrsgraph<StaticCrsGraphType> (str.append (".graph"), row_lengths);
  typename values_type::HostMirror h_values = Kokkos::create_mirror_view (values);
  typename index_type::HostMirror h_entries = Kokkos::create_mirror_view (graph.entries);

  // FIXME (mfh 21 Jun 2013) This needs to be a parallel copy.
  // Furthermore, why are the arrays copied twice? -- once here, to a
  // host view, and once below, in the deep copy?
  for (size_type i = 0; i < annz; ++i) {
    if (val) {
      h_values(i) = val[i];
    }
    h_entries(i) = cols[i];
  }

  Kokkos::deep_copy (values, h_values);
  Kokkos::deep_copy (graph.entries, h_entries);
}
}} // namespace KokkosSparse::Experimental
#endif
