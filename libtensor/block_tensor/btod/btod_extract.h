#ifndef LIBTENSOR_BTOD_EXTRACT_H
#define LIBTENSOR_BTOD_EXTRACT_H

#include <libtensor/timings.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


/** \brief Extracts a tensor with smaller dimension from the %tensor
    \tparam N Tensor order.
    \tparam M Number of fixed dimensions.
    \tparam N - M result tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N, size_t M>
class btod_extract :
    public additive_bto<N - M, btod_traits>,
    public timings< btod_extract<N, M> > {

public:
    static const char *k_clazz; //!< Class name

public:
    static const size_t k_ordera = N; //!< Order of the argument
    static const size_t k_orderb = N - M; //!< Order of the result

private:
    block_tensor_i<N, double> &m_bta; //!< Input block %tensor
    mask<N> m_msk;//!< Mask for extraction
    permutation<k_orderb> m_perm; //!< Permutation of the result
    double m_c; //!< Scaling coefficient
    block_index_space<k_orderb> m_bis; //!< Block %index space of the result
    index<N> m_idxbl;//!< Index for extraction of the block
    index<N> m_idxibl;//!< Index for extraction inside the block
    symmetry<k_orderb, double> m_sym; //!< Symmetry of the result
    assignment_schedule<k_orderb, double> m_sch; //!< Assignment schedule

public:
    btod_extract(block_tensor_i<N, double> &bta, const mask<N> &m,
        const index<N> &idxbl, const index<N> &idxibl, double c = 1.0);

    btod_extract(block_tensor_i<N, double> &bta, const mask<N> &m,
        const permutation<N - M> &perm, const index<N> &idxbl,
        const index<N> &idxibl, double c = 1.0);

    //!    \name Implementation of
    //      libtensor::direct_tensor_operation<N - M + 1, double>
    //@{

    virtual const block_index_space<k_orderb> &get_bis() const {
        return m_bis;
    }

    virtual const symmetry<k_orderb, double> &get_symmetry() const {
        return m_sym;
    }

    virtual const assignment_schedule<k_orderb, double> &get_schedule() const {
        return m_sch;
    }

    //@}

    virtual void sync_on();
    virtual void sync_off();

    virtual void perform(bto_stream_i<N - M, btod_traits> &out);
    virtual void perform(block_tensor_i<N - M, double> &btb);
    virtual void perform(block_tensor_i<N - M, double> &btb, const double &c);

    virtual void compute_block(bool zero,
        dense_tensor_i<k_orderb, double> &blk, const index<k_orderb> &i,
        const tensor_transf<k_orderb, double> &tr, const double &c);

private:
    /** \brief Forms the block %index space of the output or throws an
            exception if the input is incorrect
     **/
    static block_index_space<N - M> mk_bis(const block_index_space<N> &bis,
        const mask<N> &msk, const permutation<N - M> &perm);

    /** \brief Sets up the assignment schedule for the operation.
     **/
    void make_schedule();

    void do_compute_block(dense_tensor_i<k_orderb, double> &blk,
        const index<k_orderb> &i, const tensor_transf<k_orderb, double> &tr,
        double c, bool zero);

private:
    btod_extract(const btod_extract<N, M>&);
    const btod_extract<N, M> &operator=(const btod_extract<N, M>&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EXTRACT_H
