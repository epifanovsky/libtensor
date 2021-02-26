#ifndef LIBTENSOR_BTO_IMPORT_RAW_STREAM_H
#define LIBTENSOR_BTO_IMPORT_RAW_STREAM_H

#include <iostream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/bad_block_index_space.h>
#include <libtensor/dense_tensor/to_import_raw_stream.h>
#include "bto_import_raw_base.h"

namespace libtensor {


/** \brief Imports block tensor data from an input stream
    \tparam N Tensor order.
    \tparam Alloc Allocator for temporary buffers.

    This operation reads tensor elements from a binary input stream in which
    they are arranged in the regular tensor format. The stream must be opened
    before calling this operation.

    The symmetry of the block tensor is guessed from the initial symmetry
    and verified. If the actual data do not have the symmetry, an exception
    is raised.

    <b>Example</b>

    \code
    block_tensor<4, T, ...> bt(...);
    ifstream fs("tensor_data_file", ios::in | ios::binary);
    bto_import_raw_stream<4>(fs, bt.get_bis().get_dims(), 1e-12).
        perform(bt);
    fs.close();
    \endcode

    \sa tod_import_raw_stream<N>, bto_read<N, Alloc>,
        bto_import_raw<N, Alloc>, bto_import_raw_base<N, Alloc>

    \ingroup libtensor_btod
 **/
template<size_t N, typename T, typename Alloc = allocator<T> >
class bto_import_raw_stream : public bto_import_raw_base<N, T, Alloc> {
public:
    static const char *k_clazz; //!< Class name

private:
    std::istream &m_is; //!< Input stream
    dimensions<N> m_dims; //!< Dimensions of the memory block
    T m_zero_thresh; //!< Zero threshold
    T m_sym_thresh; //!< Symmetry threshold

public:
    /** \brief Initializes the operation
        \param is Input stream.
        \param dims Dimensions of the input.
        \param thresh Threshold for zeros and symmetry detection
     **/
    bto_import_raw_stream(std::istream &is, const dimensions<N> &dims,
        T thresh = 0.0) :
        m_is(is), m_dims(dims), m_zero_thresh(thresh), m_sym_thresh(thresh) {
    }

    /** \brief Initializes the operation
        \param is Input stream.
        \param dims Dimensions of the input.
        \param zero_thresh Threshold for zeros
        \param sym_thresh Threshold for %symmetry
     **/
    bto_import_raw_stream(std::istream &is, const dimensions<N> &dims,
        T zero_thresh, T sym_thresh) :
        m_is(is), m_dims(dims), m_zero_thresh(zero_thresh),
            m_sym_thresh(sym_thresh) {
    }

    /** \brief Performs the operation
        \param bt Output block %tensor.
     **/
    void perform(block_tensor_i<N, T> &bt);

protected:
    using bto_import_raw_base<N, T, Alloc>::verify_and_set_symmetry;
    using bto_import_raw_base<N, T, Alloc>::check_zero;

private:
    bto_import_raw_stream(const bto_import_raw_stream<N, T, Alloc>&);
    const bto_import_raw_stream<N, T, Alloc>
        &operator=(const bto_import_raw_stream<N, T, Alloc>&);

};


template<size_t N, typename T, typename Alloc>
const char *bto_import_raw_stream<N, T, Alloc>::k_clazz =
    "bto_import_raw_stream<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
void bto_import_raw_stream<N, T, Alloc>::perform(block_tensor_i<N, T> &bt) {

    static const char *method = "perform(block_tensor_i<N>&)";

    //
    //  Check the block tensor's dimensions
    //

    const block_index_space<N> &bis = bt.get_bis();
    dimensions<N> bidims(bis.get_block_index_dims());
    if(!bis.get_dims().equals(m_dims)) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "bt");
    }

    //  Set up the block tensor

    block_tensor_ctrl<N, T> ctrl(bt);
    symmetry<N, T> sym(bis);
    so_copy<N, T> (ctrl.req_const_symmetry()).perform(sym);
    ctrl.req_symmetry().clear();
    ctrl.req_zero_all_blocks();

    //  Invoke the import operation for each block

    dimensions<N> bdims(bis.get_block_index_dims());
    abs_index<N> bi(bdims);

    do {
        {
            dense_tensor_wr_i<N, T> &blk = ctrl.req_block(bi.get_index());

            index<N> blk_start(bis.get_block_start(bi.get_index()));
            dimensions<N> blk_dims(bis.get_block_dims(bi.get_index()));
            index<N> blk_end(blk_start);
            for(size_t i = 0; i < N; i++) blk_end[i] += blk_dims[i] - 1;
            index_range<N> ir(blk_start, blk_end);

            to_import_raw_stream<N, T>(m_is, m_dims, ir).perform(blk);
            ctrl.ret_block(bi.get_index());
        }

        bool zero;
        {
            dense_tensor_rd_i<N, T> &blk =
                    ctrl.req_const_block(bi.get_index());
            zero = check_zero(blk, m_zero_thresh);
            ctrl.ret_const_block(bi.get_index());
        }

        if(zero) {
            ctrl.req_zero_block(bi.get_index());
        }
    } while(bi.inc());

    verify_and_set_symmetry(bt, sym, m_sym_thresh);
}

template<size_t N, typename Alloc = allocator<double> >
using btod_import_raw_stream = bto_import_raw_stream<N, double, Alloc>; 

} // namespace libtensor

#endif // LIBTENSOR_BTO_IMPORT_RAW_STREAM_H
