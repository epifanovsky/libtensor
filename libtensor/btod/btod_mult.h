#ifndef LIBTENSOR_BTOD_MULT_H
#define LIBTENSOR_BTOD_MULT_H

#include "../defs.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../core/permutation_builder.h"
#include "../symmetry/so_dirprod.h"
#include "../symmetry/so_merge.h"
#include <libtensor/dense_tensor/tod_mult.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include "bad_block_index_space.h"

namespace libtensor {


/** \brief Elementwise multiplication of two block tensors
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_mult :
    public additive_bto<N, bto_traits<double> >,
    public timings< btod_mult<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    block_tensor_i<N, double> &m_bta; //!< First argument
    block_tensor_i<N, double> &m_btb; //!< Second argument
    permutation<N> m_pa; //!< Permutation of bta
    permutation<N> m_pb; //!< Permutation of btb
    bool m_recip; //!< Reciprocal
    double m_c; //!< Scaling coefficient

    block_index_space<N> m_bis; //!< Block %index space of the result
    symmetry<N, double> m_sym; //!< Result symmetry
    assignment_schedule<N, double> m_sch; //!< Schedule

public:
    //! \name Constructors / destructor
    //@{

    /** \brief Constructor
        \param bta First argument
        \param btb Second argument
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param c Coefficient
     **/
    btod_mult(block_tensor_i<N, double> &bta, block_tensor_i<N, double> &btb,
            bool recip = false, double c = 1.0);

    /** \brief Constructor
        \param bta First argument
        \param pa Permutation of first argument
        \param btb Second argument
        \param pb Permutation of second argument
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param c Coefficient
     **/
    btod_mult(block_tensor_i<N, double> &bta, const permutation<N> &pa,
            block_tensor_i<N, double> &btb, const permutation<N> &pb,
            bool recip = false, double c = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~btod_mult();

    //@}



    //!    \name Implementation of
    //      libtensor::direct_block_tensor_operation<N, double>
    //@{
    virtual const block_index_space<N> &get_bis() const {
        return m_bta.get_bis();
    }

    virtual const symmetry<N, double> &get_symmetry() const {
        return m_sym;
    }

    virtual const assignment_schedule<N, double> &get_schedule() const {
        return m_sch;
    }

    virtual void sync_on();
    virtual void sync_off();

    //@}

    using additive_bto<N, bto_traits<double> >::perform;

protected:
    virtual void compute_block(bool zero, dense_tensor_i<N, double> &blk,
            const index<N> &idx, const tensor_transf<N, double> &tr,
            const double &c, cpu_pool &cpus);

private:
    btod_mult(const btod_mult<N> &);
    const btod_mult<N> &operator=(const btod_mult<N> &);

    void make_schedule();
};


template<size_t N>
const char *btod_mult<N>::k_clazz = "btod_mult<N>";


template<size_t N>
btod_mult<N>::btod_mult(block_tensor_i<N, double> &bta,
    block_tensor_i<N, double> &btb, bool recip, double c) :

    m_bta(bta), m_btb(btb), m_recip(recip), m_c(c), m_bis(m_bta.get_bis()),
    m_sym(m_bta.get_bis()), m_sch(m_bta.get_bis().get_block_index_dims()) {

    static const char *method = "btod_mult(block_tensor_i<N, double>&, "
        "block_tensor_i<N, double>&, bool)";

    if(! m_bta.get_bis().equals(m_btb.get_bis())) {
        throw bad_block_index_space(g_ns, k_clazz, method,
            __FILE__, __LINE__, "bta,btb");
    }

    block_tensor_ctrl<N, double> cbta(bta), cbtb(btb);
    sequence<N + N, size_t> seq1b, seq2b;
    for (size_t i = 0; i < N; i++) {
        seq1b[i] = i; seq2b[i] = m_pa[i];
    }
    for (size_t i = N, j = 0; i < N + N; i++, j++) {
        seq1b[i] = i; seq2b[i] = m_pb[j] + N;
    }
    permutation_builder<N + N> pbb(seq2b, seq1b);

    block_index_space_product_builder<N, N> bbx(m_bis, m_bis,
            permutation<N + N>());

    symmetry<N + N, double> symx(bbx.get_bis());
    so_dirprod<N, N, double>(cbta.req_const_symmetry(),
            cbtb.req_const_symmetry(), pbb.get_perm()).perform(symx);
    mask<N + N> msk;
    sequence<N + N, size_t> seq;
    for (size_t i = 0; i < N; i++) {
        msk[i] = msk[i + N] = true;
        seq[i] = seq[i + N] = i;
    }
    so_merge<N + N, N, double>(symx, msk, seq).perform(m_sym);

    make_schedule();
}

template<size_t N>
btod_mult<N>::btod_mult(
    block_tensor_i<N, double> &bta, const permutation<N> &pa,
    block_tensor_i<N, double> &btb, const permutation<N> &pb,
    bool recip, double c) :

    m_bta(bta), m_btb(btb), m_pa(pa), m_pb(pb), m_recip(recip), m_c(c),
    m_bis(block_index_space<N>(m_bta.get_bis()).permute(m_pa)),
    m_sym(m_bis), m_sch(m_bis.get_block_index_dims()) {

    static const char *method = "btod_mult(block_tensor_i<N, double>&, "
        "block_tensor_i<N, double>&, bool)";

    block_index_space<N> bisb(m_btb.get_bis());
    bisb.permute(m_pb);
    if(! m_bis.equals(bisb)) {
        throw bad_block_index_space(g_ns, k_clazz, method,
            __FILE__, __LINE__, "bta,btb");
    }

    block_tensor_ctrl<N, double> cbta(bta), cbtb(btb);

    sequence<N + N, size_t> seq1b, seq2b;
    for (size_t i = 0; i < N; i++) {
        seq1b[i] = i; seq2b[i] = m_pa[i];
    }
    for (size_t i = N, j = 0; i < N + N; i++, j++) {
        seq1b[i] = i; seq2b[i] = m_pb[j] + N;
    }
    permutation_builder<N + N> pbb(seq2b, seq1b);

    block_index_space_product_builder<N, N> bbx(m_bis, m_bis,
            permutation<N + N>());

    symmetry<N + N, double> symx(bbx.get_bis());
    so_dirprod<N, N, double>(cbta.req_const_symmetry(),
            cbtb.req_const_symmetry(), pbb.get_perm()).perform(symx);
    mask<N + N> msk;
    sequence<N + N, size_t> seq;
    for (register size_t i = 0; i < N; i++) {
        msk[i] = msk[i + N] = true;
        seq[i] = seq[i + N] = i;
    }
    so_merge<N + N, N, double>(symx, msk, seq).perform(m_sym);

    make_schedule();
}

template<size_t N>
btod_mult<N>::~btod_mult() {

}


template<size_t N>
void btod_mult<N>::sync_on() {

    block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(m_btb);
    ctrla.req_sync_on();
    ctrlb.req_sync_on();
}


template<size_t N>
void btod_mult<N>::sync_off() {

    block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(m_btb);
    ctrla.req_sync_off();
    ctrlb.req_sync_off();
}

/*
template<size_t N>
void btod_mult<N>::compute_block(
        dense_tensor_i<N, double> &blk, const index<N> &idx) {

    block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(m_btb);

    permutation<N> pinva(m_pa, true), pinvb(m_pb, true);
    index<N> idxa(idx), idxb(idx);
    idxa.permute(pinva);
    idxb.permute(pinvb);

    orbit<N, double> oa(ctrla.req_const_symmetry(), idxa);
    abs_index<N> cidxa(oa.get_abs_canonical_index(),
            m_bta.get_bis().get_block_index_dims());
    const tensor_transf<N, double> &tra = oa.get_transf(idxa);

    orbit<N, double> ob(ctrlb.req_const_symmetry(), idxb);
    abs_index<N> cidxb(ob.get_abs_canonical_index(),
            m_btb.get_bis().get_block_index_dims());
    const tensor_transf<N, double> &trb = ob.get_transf(idxb);

    permutation<N> pa(tra.get_perm());
    pa.permute(m_pa);
    permutation<N> pb(trb.get_perm());
    pb.permute(m_pb);

    dense_tensor_i<N, double> &blka = ctrla.req_block(cidxa.get_index());
    dense_tensor_i<N, double> &blkb = ctrlb.req_block(cidxb.get_index());

    double k = m_c * tra.get_coeff();
    if (m_recip)
        k /= trb.get_coeff();
    else
        k *= trb.get_coeff();

    tod_mult<N>(blka, pa, blkb, pb, m_recip, k).perform(blk);

    ctrla.ret_block(cidxa.get_index());
    ctrlb.ret_block(cidxb.get_index());
}*/



template<size_t N>
void btod_mult<N>::compute_block(bool zero, dense_tensor_i<N, double> &blk,
        const index<N> &idx, const tensor_transf<N, double> &tr,
        const double &c, cpu_pool &cpus) {

    block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(m_btb);

    permutation<N> pinva(m_pa, true), pinvb(m_pb, true), pinvc(tr.get_perm(), true);
    index<N> idxa(idx), idxb(idx);
    idxa.permute(pinva);
    idxb.permute(pinvb);

    orbit<N, double> oa(ctrla.req_const_symmetry(), idxa);
    abs_index<N> cidxa(oa.get_abs_canonical_index(),
            m_bta.get_bis().get_block_index_dims());
    const tensor_transf<N, double> &tra = oa.get_transf(idxa);

    orbit<N, double> ob(ctrlb.req_const_symmetry(), idxb);
    abs_index<N> cidxb(ob.get_abs_canonical_index(),
            m_btb.get_bis().get_block_index_dims());
    const tensor_transf<N, double> &trb = ob.get_transf(idxb);

    permutation<N> pa(tra.get_perm());
    pa.permute(m_pa);
    pa.permute(pinvc);
    permutation<N> pb(trb.get_perm());
    pb.permute(m_pb);
    pb.permute(pinvc);

    dense_tensor_i<N, double> &blka = ctrla.req_block(cidxa.get_index());
    dense_tensor_i<N, double> &blkb = ctrlb.req_block(cidxb.get_index());

    double k = m_c * tr.get_scalar_tr().get_coeff() *
            tra.get_scalar_tr().get_coeff();
    if (m_recip)
        k /= trb.get_scalar_tr().get_coeff();
    else
        k *= trb.get_scalar_tr().get_coeff();

    if(zero) tod_set<N>().perform(cpus, blk);
    tod_mult<N>(blka, pa, blkb, pb, m_recip, k).perform(cpus, false, c, blk);

    ctrla.ret_block(cidxa.get_index());
    ctrlb.ret_block(cidxb.get_index());
}

template<size_t N>
void btod_mult<N>::make_schedule() {

    static const char *method = "make_schedule()";

    block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(m_btb);

    orbit_list<N, double> ol(m_sym);

    for (typename orbit_list<N, double>::iterator iol = ol.begin();
            iol != ol.end(); iol++) {

        index<N> idx(ol.get_index(iol));
        index<N> idxa(idx), idxb(idx);
        permutation<N> pinva(m_pa, true), pinvb(m_pb, true);
        idxa.permute(pinva);
        idxb.permute(pinvb);

        orbit<N, double> oa(ctrla.req_const_symmetry(), idxa);
        if (! oa.is_allowed())
            continue;
        abs_index<N> cidxa(oa.get_abs_canonical_index(),
                m_bta.get_bis().get_block_index_dims());
        bool zeroa = ctrla.req_is_zero_block(cidxa.get_index());

        orbit<N, double> ob(ctrlb.req_const_symmetry(), idxb);
        if (! ob.is_allowed()) {
            if (m_recip)
                throw bad_parameter(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "Block not allowed in btb.");

            continue;
        }

        abs_index<N> cidxb(ob.get_abs_canonical_index(),
                m_btb.get_bis().get_block_index_dims());
        bool zerob = ctrlb.req_is_zero_block(cidxb.get_index());

        if (m_recip && zerob) {
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "zero in btb");
        }

        if (! zeroa && ! zerob) {
            m_sch.insert(idx);
        }

    }


}




} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT_H
