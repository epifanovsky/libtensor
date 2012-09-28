#ifndef LIBTENSOR_BTOD_EXTRACT_H
#define LIBTENSOR_BTOD_EXTRACT_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_extract.h>

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
    public noncopyable {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename btod_traits::bti_traits bti_traits;
    typedef tensor_transf<N - M, double> tensor_transf_type;

private:
    gen_bto_extract< N, M, btod_traits, btod_extract<N, M> > m_gbto;

public:
    btod_extract(
        block_tensor_rd_i<N, double> &bta, const mask<N> &m,
        const index<N> &idxbl, const index<N> &idxibl,
        const tensor_transf_type &trb = tensor_transf_type()) :
        m_gbto(bta, m, idxbl, idxibl, trb) {
    }

    btod_extract(
        block_tensor_rd_i<N, double> &bta, const mask<N> &m,
        const index<N> &idxbl, const index<N> &idxibl, double c) :
        m_gbto(bta, m, idxbl, idxibl, tensor_transf_type(permutation<N - M>(),
                scalar_transf<double>(c))) {

    }

    btod_extract(
        block_tensor_rd_i<N, double> &bta, const mask<N> &m,
        const index<N> &idxbl, const index<N> &idxibl,
        const permutation<N - M> &perm, double c = 1.0) :
        m_gbto(bta, m, idxbl, idxibl,
                tensor_transf_type(perm, scalar_transf<double>(c))) {

    }

    /** \brief Virtual destructor
     **/
    virtual ~btod_extract() { }

    //!    \name Implementation of
    //      libtensor::direct_tensor_operation<N - M + 1, double>
    //@{

    virtual const block_index_space<N - M> &get_bis() const {
        return m_gbto.get_bis();
    }

    virtual const symmetry<N - M, double> &get_symmetry() const {
        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N - M, double> &get_schedule() const {
        return m_gbto.get_schedule();
    }

    //@}

    virtual void perform(gen_block_stream_i<N - M, bti_traits> &out) {
        m_gbto.perform(out);
    }

    virtual void perform(block_tensor_i<N - M, double> &btb);
    virtual void perform(block_tensor_i<N - M, double> &btb, const double &c);

    virtual void compute_block(dense_tensor_i<N - M, double> &blkb,
            const index<N - M> &ib);

    virtual void compute_block(bool zero,
            dense_tensor_i<N - M, double> &blkb, const index<N - M> &ib,
            const tensor_transf<N - M, double> &trb, const double &c);
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EXTRACT_H
