#ifndef LIBTENSOR_BTENSOR_RENDERER_DIRSUM_H
#define LIBTENSOR_BTENSOR_RENDERER_DIRSUM_H

#include <libtensor/core/permutation_builder.h>
#include <libtensor/expr/dirsum/expression_node_dirsum.h>
#include <libtensor/expr/ident/expression_node_ident.h>
#include <libtensor/btod/btod_dirsum.h>
#include "../btensor.h"
#include "../btensor_renderer.h"
#include "btensor_operation_container_i.h"

namespace libtensor {


/** \brief Contains rendered expression_node_dirsum

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
class btensor_operation_container_dirsum;


/** \brief Renders expression_node_dirsum into btensor_i

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
class btensor_renderer_dirsum;


/** \brief Contains rendered expression_node_dirsum (specialized for double)

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M>
class btensor_operation_container_dirsum<N, M, double> :
    public btensor_operation_container_i<N + M, double> {

private:
    std::auto_ptr< btensor_operation_container_i<N, double> > m_exa;
    std::auto_ptr< btensor_operation_container_i<M, double> > m_exb;
    btensor_i<N, double> *m_bta;
    btensor_i<M, double> *m_btb;
    double m_sa;
    double m_sb;
    permutation<N + M> m_permc;

public:
    /** \brief Initializes the container with tensors A and B
     **/
    btensor_operation_container_dirsum(btensor_i<N, double> &bta, double sa,
        btensor_i<M, double> &btb, double sb, const permutation<N + M> &permc) :
        m_bta(&bta), m_sa(sa), m_btb(&btb), m_sb(sb), m_permc(permc) { }

    /** \brief Initializes the container with tensor A and operation B
     **/
    btensor_operation_container_dirsum(btensor_i<N, double> &bta, double sa,
        std::auto_ptr< btensor_operation_container_i<M, double> > &exb,
        double sb, const permutation<N + M> &permc) :
        m_exb(exb), m_bta(&bta), m_sa(sa), m_btb(0), m_sb(sb), m_permc(permc)
        { }

    /** \brief Initializes the container with operation A and tensor B
     **/
    btensor_operation_container_dirsum(
        std::auto_ptr< btensor_operation_container_i<N, double> > &exa,
        double sa, btensor_i<M, double> &btb, double sb,
        const permutation<N + M> &permc) :
        m_exa(exa), m_bta(0), m_sa(sa), m_btb(&btb), m_sb(sb), m_permc(permc)
        { }

    /** \brief Initializes the container with operations A and B
     **/
    btensor_operation_container_dirsum(
        std::auto_ptr< btensor_operation_container_i<N, double> > &exa,
        double sa,
        std::auto_ptr< btensor_operation_container_i<M, double> > &exb,
        double sb, const permutation<N + M> &permc) :
        m_exa(exa), m_exb(exb), m_bta(0), m_sa(sa), m_btb(0), m_sb(sb),
        m_permc(permc) { }

    /** \brief Destroys the container
     **/
    virtual ~btensor_operation_container_dirsum() {

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

        btod_dirsum<N, M> op(*pbta, m_sa, *pbtb, m_sb, m_permc);
        if(add) op.perform(bt, 1.0);
        else op.perform(bt);
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

        btod_dirsum<N, M> op(*pbta, m_sa, *pbtb, m_sb, m_permc);
        std::auto_ptr< btensor_i<N + M, double> > bt(
            new btensor<N + M, double>(op.get_bis()));
        op.perform(*bt);

        return bt;
    }

};


/** \brief Renders expression_node_dirsum into btensor_i (specialized
        for double)

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M>
class btensor_renderer_dirsum<N, M, double> {
public:
    std::auto_ptr< btensor_operation_container_i<N + M, double> > render_node(
        expression_node_dirsum<N, M, double> &n);

};


template<size_t N, size_t M>
std::auto_ptr< btensor_operation_container_i<N + M, double> >
btensor_renderer_dirsum<N, M, double>::render_node(
    expression_node_dirsum<N, M, double> &n) {

    std::auto_ptr< btensor_operation_container_i<N, double> > exa;
    std::auto_ptr< btensor_operation_container_i<M, double> > exb;
    btensor_i<N, double> *bta = 0;
    btensor_i<M, double> *btb = 0;

    sequence<N + M, size_t> seq1(0), seq2(0);
    for(size_t i = 0; i < N + M; i++) seq1[i] = seq2[i] = i;
    n.get_perm().apply(seq2);
    double sa(n.get_s()), sb(n.get_s());

    const std::vector< expression_node<N, double>* > &nla =
        n.get_a().get_nodes();
    const std::vector< expression_node<M, double>* > &nlb =
        n.get_b().get_nodes();

    if(nla.size() == 1 &&
        expression_node_ident<N, double>::check_type(*nla[0])) {

        expression_node_ident<N, double> &na =
            expression_node_ident<N, double>::cast(*nla[0]);
        bta = &dynamic_cast<btensor_i<N, double>&>(na.get_tensor());
        sa *= na.get_s();
        if(!na.get_perm().is_identity()) {
            sequence<N, size_t> seq3(0);
            for(size_t i = 0; i < N; i++) seq3[i] = seq1[i];
            permutation<N>(na.get_perm(), true).apply(seq3);
            for(size_t i = 0; i < N; i++) seq1[i] = seq3[i];
        }

    } else {

        exa = btensor_renderer<N, double>().render(n.get_a());
    }

    if(nlb.size() == 1 &&
        expression_node_ident<M, double>::check_type(*nlb[0])) {

        expression_node_ident<M, double> &nb =
            expression_node_ident<M, double>::cast(*nlb[0]);
        btb = &dynamic_cast<btensor_i<M, double>&>(nb.get_tensor());
        sb *= nb.get_s();
        if(!nb.get_perm().is_identity()) {
            sequence<M, size_t> seq3(0);
            for(size_t i = 0; i < M; i++) seq3[i] = seq1[N + i];
            permutation<M>(nb.get_perm(), true).apply(seq3);
            for(size_t i = 0; i < M; i++) seq1[N + i] = seq3[i];
        }

    } else {

        exb = btensor_renderer<M, double>().render(n.get_b());
    }


    permutation_builder<N + M> pbc(seq1, seq2);
    btensor_operation_container_i<N + M, double> *boc = 0;

    if(bta && btb) {

        boc = new btensor_operation_container_dirsum<N, M, double>(
            *bta, sa, *btb, sb, pbc.get_perm());

    } else if(bta) {

        boc = new btensor_operation_container_dirsum<N, M, double>(
            *bta, sa, exb, sb, pbc.get_perm());

    } else if(btb) {

        boc = new btensor_operation_container_dirsum<N, M, double>(
            exa, sa, *btb, sb, pbc.get_perm());

    } else {

        boc = new btensor_operation_container_dirsum<N, M, double>(
            exa, sa, exb, sb, pbc.get_perm());

    }

    return std::auto_ptr< btensor_operation_container_i<N + M, double> >(boc);
}


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_RENDERER_DIRSUM_H
