#ifndef LIBTENSOR_BTENSOR_RENDERER_DIRPROD_H
#define LIBTENSOR_BTENSOR_RENDERER_DIRPROD_H

#include <libtensor/expr/dirprod/expression_node_dirprod.h>
#include <libtensor/expr/ident/expression_node_ident.h>
#include <libtensor/btod/btod_contract2.h>
#include "../btensor.h"
#include "../btensor_renderer.h"
#include "btensor_operation_container_i.h"

namespace libtensor {


/** \brief Contains rendered expression_node_dirprod

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
class btensor_operation_container_dirprod;


/** \brief Renders expression_node_dirprod into btensor_i

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
class btensor_renderer_dirprod;


/** \brief Contains rendered expression_node_dirprod (specialized for double)

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M>
class btensor_operation_container_dirprod<N, M, double> :
    public btensor_operation_container_i<N + M, double> {

private:
    contraction2<N, M, 0> m_contr;
    std::auto_ptr< btensor_operation_container_i<N, double> > m_exa;
    std::auto_ptr< btensor_operation_container_i<M, double> > m_exb;
    btensor_i<N, double> *m_bta;
    btensor_i<M, double> *m_btb;
    double m_s;

public:
    /** \brief Initializes the container with tensors A and B
     **/
    btensor_operation_container_dirprod(const contraction2<N, M, 0> &contr,
        btensor_i<N, double> &bta, btensor_i<M, double> &btb, double s) :
        m_contr(contr), m_bta(&bta), m_btb(&btb), m_s(s) { }

    /** \brief Initializes the container with tensor A and operation B
     **/
    btensor_operation_container_dirprod(const contraction2<N, M, 0> &contr,
        btensor_i<N, double> &bta,
        std::auto_ptr< btensor_operation_container_i<M, double> > &exb,
        double s) :
        m_contr(contr), m_exb(exb), m_bta(&bta), m_btb(0), m_s(s) { }

    /** \brief Initializes the container with operation A and tensor B
     **/
    btensor_operation_container_dirprod(const contraction2<N, M, 0> &contr,
        std::auto_ptr< btensor_operation_container_i<N, double> > &exa,
        btensor_i<M, double> &btb, double s) :
        m_contr(contr), m_exa(exa), m_bta(0), m_btb(&btb), m_s(s) { }

    /** \brief Initializes the container with operations A and B
     **/
    btensor_operation_container_dirprod(const contraction2<N, M, 0> &contr,
        std::auto_ptr< btensor_operation_container_i<N, double> > &exa,
        std::auto_ptr< btensor_operation_container_i<M, double> > &exb,
        double s) :
        m_contr(contr), m_exa(exa), m_exb(exb), m_bta(0), m_btb(0), m_s(s) { }

    /** \brief Destroys the container
     **/
    virtual ~btensor_operation_container_dirprod() {

    }

    virtual void perform(bool add, btensor_i<N + M, double> &bt) {

        std::auto_ptr< btensor_i<N, double> > bta;
        std::auto_ptr< btensor_i<M, double> > btb;

        btensor_i<N, double> *pbta;
        btensor_i<M, double> *pbtb;

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

        btod_contract2<N, M, 0> op(m_contr, *pbta, *pbtb);
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

        std::auto_ptr< btensor_i<N, double> > bta;
        std::auto_ptr< btensor_i<M, double> > btb;

        btensor_i<N, double> *pbta;
        btensor_i<M, double> *pbtb;

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

        btod_contract2<N, M, 0> op(m_contr, *pbta, *pbtb);
        std::auto_ptr< btensor_i<N + M, double> > bt(
            new btensor<N + M, double>(op.get_bis()));
        op.perform(*bt);
        btod_scale<N + M>(*bt, m_s).perform();

        return bt;
    }

};


/** \brief Renders expression_node_dirprod into btensor_i (specialized
        for double)

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M>
class btensor_renderer_dirprod<N, M, double> {
public:
    std::auto_ptr< btensor_operation_container_i<N + M, double> > render_node(
        expression_node_dirprod<N, M, double> &n);

};


template<size_t N, size_t M>
std::auto_ptr< btensor_operation_container_i<N + M, double> >
btensor_renderer_dirprod<N, M, double>::render_node(
    expression_node_dirprod<N, M, double> &n) {

    std::auto_ptr< btensor_operation_container_i<N, double> > exa;
    std::auto_ptr< btensor_operation_container_i<M, double> > exb;
    btensor_i<N, double> *bta = 0;
    btensor_i<M, double> *btb = 0;

    contraction2<N, M, 0> contr(n.get_perm());
    double s(n.get_s());

    const std::vector< expression_node<N, double>* > &nla =
        n.get_a().get_nodes();
    const std::vector< expression_node<M, double>* > &nlb =
        n.get_b().get_nodes();

    if(nla.size() == 1 &&
        expression_node_ident<N, double>::check_type(*nla[0])) {

        expression_node_ident<N, double> &na =
            expression_node_ident<N, double>::cast(*nla[0]);
        bta = &dynamic_cast<btensor_i<N, double>&>(na.get_tensor());
        s *= na.get_s();
        contr.permute_a(na.get_perm());

    } else {

        exa = btensor_renderer<N, double>().render(n.get_a());
    }

    if(nlb.size() == 1 &&
        expression_node_ident<M, double>::check_type(*nlb[0])) {

        expression_node_ident<M, double> &nb =
            expression_node_ident<M, double>::cast(*nlb[0]);
        btb = &dynamic_cast<btensor_i<M, double>&>(nb.get_tensor());
        s *= nb.get_s();
        contr.permute_b(nb.get_perm());

    } else {

        exb = btensor_renderer<M, double>().render(n.get_b());
    }


    btensor_operation_container_i<N + M, double> *boc = 0;

    if(bta && btb) {

        boc = new btensor_operation_container_dirprod<N, M, double>(
            contr, *bta, *btb, s);

    } else if(bta) {

        boc = new btensor_operation_container_dirprod<N, M, double>(
            contr, *bta, exb, s);

    } else if(btb) {

        boc = new btensor_operation_container_dirprod<N, M, double>(
            contr, exa, *btb, s);

    } else {

        boc = new btensor_operation_container_dirprod<N, M, double>(
            contr, exa, exb, s);

    }

    return std::auto_ptr< btensor_operation_container_i<N + M, double> >(boc);
}


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_RENDERER_DIRPROD_H
