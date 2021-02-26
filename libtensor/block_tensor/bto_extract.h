#ifndef LIBTENSOR_BTO_EXTRACT_H
#define LIBTENSOR_BTO_EXTRACT_H

#include <libtensor/block_tensor/bto_traits.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_extract.h>

namespace libtensor {


/** \brief Extracts a tensor with smaller dimension from the %tensor
    \tparam N Tensor order.
    \tparam M Number of fixed dimensions.
    \tparam N - M result tensor order.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, size_t M, typename T>
class bto_extract :
    public additive_gen_bto<N - M, typename bto_traits<T>::bti_traits>,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename bto_traits<T>::bti_traits bti_traits;
    typedef tensor_transf<N - M, T> tensor_transf_type;

private:
    gen_bto_extract< N, M, bto_traits<T>, bto_extract<N, M, T> > m_gbto;

public:
    bto_extract(
        block_tensor_rd_i<N, T> &bta, const mask<N> &m,
        const index<N> &idxbl, const index<N> &idxibl,
        const tensor_transf_type &trb = tensor_transf_type()) :
        m_gbto(bta, m, idxbl, idxibl, trb) {
    }

    bto_extract(
        block_tensor_rd_i<N, T> &bta, const mask<N> &m,
        const index<N> &idxbl, const index<N> &idxibl, T c) :
        m_gbto(bta, m, idxbl, idxibl, tensor_transf_type(permutation<N - M>(),
                scalar_transf<T>(c))) {

    }

    bto_extract(
        block_tensor_rd_i<N, T> &bta, const mask<N> &m,
        const index<N> &idxbl, const index<N> &idxibl,
        const permutation<N - M> &perm, T c = 1.0) :
        m_gbto(bta, m, idxbl, idxibl,
                tensor_transf_type(perm, scalar_transf<T>(c))) {

    }

    /** \brief Virtual destructor
     **/
    virtual ~bto_extract() { }

    //! \name Implementation of libtensor::direct_gen_bto<N - M, bti_traits>
    //@{

    virtual const block_index_space<N - M> &get_bis() const {
        return m_gbto.get_bis();
    }

    virtual const symmetry<N - M, T> &get_symmetry() const {
        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N - M, T> &get_schedule() const {
        return m_gbto.get_schedule();
    }

    virtual void perform(gen_block_stream_i<N - M, bti_traits> &out) {
        m_gbto.perform(out);
    }

    //@}

    //! \name Implementation of libtensor::additive_gen_bto<N - M, bti_traits>
    //@{

    virtual void perform(gen_block_tensor_i<N - M, bti_traits> &btb);

    virtual void perform(gen_block_tensor_i<N - M, bti_traits> &btb,
            const scalar_transf<T> &c);

    virtual void compute_block(
            bool zero,
            const index<N - M> &ib,
            const tensor_transf<N - M, T> &trb,
            dense_tensor_wr_i<N - M, T> &blkb);

    virtual void compute_block(
            const index<N - M> &ib,
            dense_tensor_wr_i<N - M, T> &blkb) {

        compute_block(true, ib, tensor_transf<N - M, T>(), blkb);
    }

    //@}

    void perform(block_tensor_i<N - M, T> &btb, T c);
};


} // namespace libtensor

#endif // LIBTENSOR_BTO_EXTRACT_H
