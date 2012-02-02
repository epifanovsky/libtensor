#ifndef LIBTENSOR_BTENSOR_RENDERER_CONTRACT2_H
#define LIBTENSOR_BTENSOR_RENDERER_CONTRACT2_H

#include <libtensor/expr/contract/expression_node_contract2.h>
#include <libtensor/expr/ident/expression_node_ident.h>
#include <libtensor/btod/btod_contract2.h>
#include "../btensor.h"
#include "../btensor_renderer.h"
#include "btensor_operation_container_i.h"

namespace libtensor {


/** \brief Contains rendered expression_node_contract2

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class btensor_operation_container_contract2;


/** \brief Renders expression_node_contract2 into btensor_i

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class btensor_renderer_contract2;


/** \brief Contains rendered expression_node_contract2 (specialized for double)

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, size_t K>
class btensor_operation_container_contract2<N, M, K, double> :
    public btensor_operation_container_i<N + M, double> {

private:
    contraction2<N, M, K> m_contr;
    std::auto_ptr< btensor_operation_container_i<N + K, double> > m_exa;
    std::auto_ptr< btensor_operation_container_i<M + K, double> > m_exb;
    btensor_i<N + K, double> *m_bta;
    btensor_i<M + K, double> *m_btb;
    double m_s;

public:
    /** \brief Initializes the container with tensors A and B
     **/
    btensor_operation_container_contract2(const contraction2<N, M, K> &contr,
        btensor_i<N + K, double> &bta, btensor_i<M + K, double> &btb,
        double s) :
        m_contr(contr), m_bta(&bta), m_btb(&btb), m_s(s) { }

    /** \brief Initializes the container with tensor A and operation B
     **/
    btensor_operation_container_contract2(const contraction2<N, M, K> &contr,
        btensor_i<N + K, double> &bta,
        std::auto_ptr< btensor_operation_container_i<M + K, double> > &exb,
        double s) :
        m_contr(contr), m_exb(exb), m_bta(&bta), m_btb(0), m_s(s) { }

    /** \brief Initializes the container with operation A and tensor B
     **/
    btensor_operation_container_contract2(const contraction2<N, M, K> &contr,
        std::auto_ptr< btensor_operation_container_i<N + K, double> > &exa,
        btensor_i<M + K, double> &btb,
        double s) :
        m_contr(contr), m_exa(exa), m_bta(0), m_btb(&btb), m_s(s) { }

    /** \brief Initializes the container with operations A and B
     **/
    btensor_operation_container_contract2(const contraction2<N, M, K> &contr,
        std::auto_ptr< btensor_operation_container_i<N + K, double> > &exa,
        std::auto_ptr< btensor_operation_container_i<M + K, double> > &exb,
        double s) :
        m_contr(contr), m_exa(exa), m_exb(exb), m_bta(0), m_btb(0), m_s(s) { }

    /** \brief Destroys the container
     **/
    virtual ~btensor_operation_container_contract2() {

    }

    virtual void perform(bool add, btensor_i<N + M, double> &bt) {

        std::auto_ptr< btensor_i<N + K, double> > bta;
        std::auto_ptr< btensor_i<M + K, double> > btb;

        btensor_i<N + K, double> *pbta;
        btensor_i<M + K, double> *pbtb;

        if(m_exa.get()) {
            bta = m_exa->perform();
            pbta = bta.get();
        } else {
            pbta = m_bta;
        }
        if(m_exb.get()) {
            btb = m_exb->perform();
            pbtb = btb.get();
        } else {
            pbtb = m_btb;
        }

        if(pbta == 0 || pbtb == 0) {
            throw 0;
        }

        btod_contract2<N, M, K> op(m_contr, *pbta, *pbtb);
        if(add) {
            op.perform(bt, m_s);
        } else {
            op.perform(bt);
            btod_scale<N + M>(bt, m_s).perform();
        }
    }

    /** \brief Performs the block tensor operation into a new tensor
     **/
    virtual std::auto_ptr< btensor_i<N + M, double> > perform() {

        std::auto_ptr< btensor_i<N + K, double> > bta;
        std::auto_ptr< btensor_i<M + K, double> > btb;

        btensor_i<N + K, double> *pbta;
        btensor_i<M + K, double> *pbtb;

        if(m_exa.get()) {
            bta = m_exa->perform();
            pbta = bta.get();
        } else {
            pbta = m_bta;
        }
        if(m_exb.get()) {
            btb = m_exb->perform();
            pbtb = btb.get();
        } else {
            pbtb = m_btb;
        }

        if(pbta == 0 || pbtb == 0) {
            throw 0;
        }

        btod_contract2<N, M, K> op(m_contr, *pbta, *pbtb);
        std::auto_ptr< btensor_i<N + M, double> > bt(
            new btensor<N + M, double>(op.get_bis()));
        op.perform(*bt);
        btod_scale<N + M>(*bt, m_s).perform();

        return bt;
    }

};


/** \brief Renders expression_node_contract2 into btensor_i (specialized
        for double)

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, size_t K>
class btensor_renderer_contract2<N, M, K, double> {
public:
    std::auto_ptr< btensor_operation_container_i<N + M, double> > render_node(
        expression_node_contract2<N, M, K, double> &n);

};


template<size_t N, size_t M, size_t K>
std::auto_ptr< btensor_operation_container_i<N + M, double> >
btensor_renderer_contract2<N, M, K, double>::render_node(
    expression_node_contract2<N, M, K, double> &n) {

    std::auto_ptr< btensor_operation_container_i<N + K, double> > exa;
    std::auto_ptr< btensor_operation_container_i<M + K, double> > exb;
    btensor_i<N + K, double> *bta = 0;
    btensor_i<M + K, double> *btb = 0;

    contraction2<N, M, K> contr(n.get_contr());
    double s(n.get_s());

    const std::vector< expression_node<N + K, double>* > &nla =
        n.get_a().get_nodes();
    const std::vector< expression_node<M + K, double>* > &nlb =
        n.get_b().get_nodes();

    if(nla.size() == 1 &&
        expression_node_ident<N + K, double>::check_type(*nla[0])) {

        expression_node_ident<N + K, double> &na =
            expression_node_ident<N + K, double>::cast(*nla[0]);
        bta = &dynamic_cast<btensor_i<N + K, double>&>(na.get_tensor());
        s *= na.get_s();
        contr.permute_a(na.get_perm());

    } else {

        exa = btensor_renderer<N + K, double>().render(n.get_a());
    }

    if(nlb.size() == 1 &&
        expression_node_ident<M + K, double>::check_type(*nlb[0])) {

        expression_node_ident<M + K, double> &nb =
            expression_node_ident<M + K, double>::cast(*nlb[0]);
        btb = &dynamic_cast<btensor_i<M + K, double>&>(nb.get_tensor());
        s *= nb.get_s();
        contr.permute_b(nb.get_perm());

    } else {

        exb = btensor_renderer<M + K, double>().render(n.get_b());
    }


    btensor_operation_container_i<N + M, double> *boc = 0;

    if(bta && btb) {

        boc = new btensor_operation_container_contract2<N, M, K, double>(
            contr, *bta, *btb, s);

    } else if(bta) {

        boc = new btensor_operation_container_contract2<N, M, K, double>(
            contr, *bta, exb, s);

    } else if(btb) {

        boc = new btensor_operation_container_contract2<N, M, K, double>(
            contr, exa, *btb, s);

    } else {

        boc = new btensor_operation_container_contract2<N, M, K, double>(
            contr, exa, exb, s);

    }

    return std::auto_ptr< btensor_operation_container_i<N + M, double> >(boc);
}


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_RENDERER_CONTRACT2_H
