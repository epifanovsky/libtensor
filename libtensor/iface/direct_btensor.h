#ifndef LIBTENSOR_DIRECT_BTENSOR_H
#define LIBTENSOR_DIRECT_BTENSOR_H

#include <memory>
#include <libtensor/exception.h>
#include <libtensor/block_tensor/direct_block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include "btensor_i.h"
#include "btensor_traits.h"
#include "labeled_btensor.h"
#include "labeled_btensor_expr.h"
#include "expr/expr.h"
#include "expr/eval_i.h"
#include "expr/evalfunctor.h"

namespace libtensor {


/** \brief User-friendly direct block %tensor

    \ingroup libtensor_iface
 **/
template<size_t N, typename T = double, typename Traits = btensor_traits<T> >
class direct_btensor : public btensor_i<N, T> {
public:
    typedef block_tensor_i_traits<T> bti_traits;

private:
    typedef struct {
        labeled_btensor_expr::expr<N, T> *m_pexpr;
        labeled_btensor_expr::eval_container_i<N, T> *m_peval;
        labeled_btensor_expr::evalfunctor_i<N, T> *m_pfunc;
    } ptrs_t;

private:
    letter_expr<N> m_label;
    ptrs_t m_ptrs;
    direct_block_tensor<N, T, typename Traits::allocator_t> m_bt;
    gen_block_tensor_rd_ctrl<N, bti_traits> m_ctrl;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates the direct block tensor using an expression
     **/
    direct_btensor(
        const letter_expr<N> &label,
        const labeled_btensor_expr::expr<N, T> &e);

    /** \brief Virtual destructor
     **/
    virtual ~direct_btensor();

    //@}

    /** \brief Attaches a label to this %tensor and returns it as a
            labeled %tensor
     **/
    labeled_btensor<N, T, false> operator()(const letter_expr<N> &expr);


    //!    \name Implementation of block_tensor_i<N, T>
    //@{

    /** \copydoc block_tensor_i<N, T>::get_bis()
     **/
    virtual const block_index_space<N> &get_bis() const;

    //@}

protected:

    //!    \name Implementation of block_tensor_rd_i<N, T>
    //@{
    virtual const symmetry<N, T> &on_req_const_symmetry();
    virtual dense_tensor_rd_i<N, T> &on_req_const_block(const index<N> &idx);
    virtual void on_ret_const_block(const index<N> &idx);
    virtual bool on_req_is_zero_block(const index<N> &idx);
    virtual void on_req_nonzero_blocks(std::vector<size_t> &nzlst);
    //@}

    //!    \name Implementation of block_tensor_wr_i<N, T>
    //@{
    virtual symmetry<N, T> &on_req_symmetry();
    virtual dense_tensor_i<N, T> &on_req_block(const index<N> &idx);
    virtual void on_ret_block(const index<N> &idx);
    virtual void on_req_zero_block(const index<N> &idx);
    virtual void on_req_zero_all_blocks();
    //@}

private:
    static ptrs_t mk_func(
        const letter_expr<N> &label,
        const labeled_btensor_expr::expr<N, T> &expr);

};


template<size_t N, typename T, typename Traits>
direct_btensor<N, T, Traits>::direct_btensor(
    const letter_expr<N> &label,
    const labeled_btensor_expr::expr<N, T> &e) :

    m_label(label), m_ptrs(mk_func(m_label, e)),
    m_bt(m_ptrs.m_pfunc->get_bto()), m_ctrl(m_bt) {

}


template<size_t N, typename T, typename Traits>
direct_btensor<N, T, Traits>::~direct_btensor() {

    delete m_ptrs.m_pfunc;
    delete m_ptrs.m_peval;
    delete m_ptrs.m_pexpr;
}


template<size_t N, typename T, typename Traits>
typename direct_btensor<N, T, Traits>::ptrs_t
direct_btensor<N, T, Traits>::mk_func(
    const letter_expr<N> &label,
    const labeled_btensor_expr::expr<N, T> &e) {

    std::auto_ptr< labeled_btensor_expr::expr<N, T> > pexpr(
        new labeled_btensor_expr::expr<N, T>(e));
    std::auto_ptr< labeled_btensor_expr::eval_container_i<N, T> > peval(
        pexpr->get_core().create_container(label));
    peval->prepare();

    ptrs_t ptrs;
    ptrs.m_pexpr = pexpr.release();
    ptrs.m_peval = peval.release();
    ptrs.m_pfunc = new labeled_btensor_expr::evalfunctor<N, T>(
        *ptrs.m_pexpr, *ptrs.m_peval);
    return ptrs;
}


template<size_t N, typename T, typename Traits>
labeled_btensor<N, T, false> direct_btensor<N, T, Traits>::operator()(
    const letter_expr<N> &label) {

    return labeled_btensor<N, T, false>(*this, label);
}


template<size_t N, typename T, typename Traits>
const block_index_space<N> &direct_btensor<N, T, Traits>::get_bis() const {

    return m_bt.get_bis();
}


template<size_t N, typename T, typename Traits>
const symmetry<N, T> &direct_btensor<N, T, Traits>::on_req_const_symmetry() {

    return m_ctrl.req_const_symmetry();
}


template<size_t N, typename T, typename Traits>
dense_tensor_rd_i<N, T> &direct_btensor<N, T, Traits>::on_req_const_block(
    const index<N> &idx) {

    return m_ctrl.req_const_block(idx);
}


template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_ret_const_block(const index<N> &idx) {

    m_ctrl.ret_const_block(idx);
}


template<size_t N, typename T, typename Traits>
bool direct_btensor<N, T, Traits>::on_req_is_zero_block(const index<N> &idx) {

    return m_ctrl.req_is_zero_block(idx);
}


template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_nonzero_blocks(
    std::vector<size_t> &nzlst) {

    m_ctrl.req_nonzero_blocks(nzlst);
}


template<size_t N, typename T, typename Traits>
symmetry<N, T> &direct_btensor<N, T, Traits>::on_req_symmetry() {

    throw immut_violation(g_ns, "direct_btensor<N, T, Traits>", "on_req_symmetry()", __FILE__, __LINE__, "");
}


template<size_t N, typename T, typename Traits>
dense_tensor_i<N, T> &direct_btensor<N, T, Traits>::on_req_block(
    const index<N> &idx) {

    throw immut_violation(g_ns, "direct_btensor<N, T, Traits>", "on_req_block()", __FILE__, __LINE__, "");
}


template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_ret_block(const index<N> &idx) {

    throw immut_violation(g_ns, "direct_btensor<N, T, Traits>", "on_ret_block()", __FILE__, __LINE__, "");
}


template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_zero_block(const index<N> &idx) {

    throw immut_violation(g_ns, "direct_btensor<N, T, Traits>", "on_req_zero_block()", __FILE__, __LINE__, "");
}


template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_zero_all_blocks() {

    throw immut_violation(g_ns, "direct_btensor<N, T, Traits>", "on_req_zero_all_blocks()", __FILE__, __LINE__, "");
}


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BTENSOR_H
