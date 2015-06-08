#include <memory>
#include <vector>
#include <libtensor/ctf_block_tensor/ctf_btod_copy.h>
#include <libtensor/ctf_block_tensor/ctf_btod_set.h>
#include <libtensor/ctf_block_tensor/ctf_btod_set_diag.h>
#include <libtensor/ctf_block_tensor/ctf_btod_shift_diag.h>
#include <libtensor/expr/dag/node_const_scalar.h>
#include <libtensor/expr/dag/node_set.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/eval/eval_exception.h>
#include <libtensor/gen_block_tensor/addition_schedule.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/symmetry/so_copy.h>
#include "ctf_btensor_from_node.h"
#include "eval_ctf_btensor_double_set.h"
#include "eval_ctf_btensor_double_autoselect.h"

namespace libtensor {
namespace expr {
namespace eval_ctf_btensor_double {

namespace {
using std::auto_ptr;


template<size_t N>
class eval_set_impl : public eval_ctf_btensor_evaluator_i<N, double> {
private:
    enum {
        Nmax = set<N>::Nmax
    };

public:
    typedef typename eval_ctf_btensor_evaluator_i<N, double>::bti_traits
        bti_traits;

private:
    ctf_btensor_i<N, double> *m_bt; //!< Tensor
    additive_gen_bto<N, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_set_impl(const expr_tree &tree, expr_tree::node_id_t id,
        const tensor_transf<N, double> &tr);

    virtual ~eval_set_impl();

    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return *m_op;
    }

private:
    static void perform_op(
        const node_set &n,
        additive_gen_bto<N, bti_traits> &bto,
        double v,
        ctf_btensor<N, double> &bt);
};


template<size_t N>
eval_set_impl<N>::eval_set_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<N, double> &tr) {

    const node_set &n = tree.get_vertex(id).template recast_as<node_set>();

    // Retrieve tensor argument and respective tensor operation
    const expr_tree::edge_list_t &e = tree.get_edges_out(id);
    tensor_transf<N, double> trx;
    expr_tree::node_id_t rhs = transf_from_node(tree, e[0], trx);

    autoselect<N> eval(tree, rhs, trx);
    additive_gen_bto<N, bti_traits> &op = eval.get_bto();

    // Retrieve scalar argument
    const node_const_scalar<double> &ns = tree.get_vertex(e[1]).
            template recast_as< node_const_scalar<double> >();
    const double &val = ns.get_scalar();

    // Create tensor
    std::auto_ptr< ctf_btensor<N, double> > bt(
            new ctf_btensor<N, double>(op.get_bis()));
    perform_op(n, op, val, *bt);

    m_bt = bt.release();
    m_op = new ctf_btod_copy<N>(*m_bt, tr.get_perm(),
            tr.get_scalar_tr().get_coeff());
}


template<size_t N>
eval_set_impl<N>::~eval_set_impl() {

    delete m_op;
    delete m_bt;
}


template<size_t N>
void eval_set_impl<N>::perform_op(
    const node_set &n,
    additive_gen_bto<N, bti_traits> &bto,
    double val,
    ctf_btensor<N, double> &bt) {

    const std::vector<size_t> &idx = n.get_idx();
    if (idx.size() != N) {
        throw eval_exception(__FILE__, __LINE__,
                "libtensor::expr::eval_ctf_btensor_double", "eval_set_impl<N>",
                "perform_op()", "Number of tensor indexes");
    }

    std::set<size_t> found;
    std::map<size_t, size_t> diagidx;
    for (size_t i = 0, d = 1; i < N; i++) {
        if (found.find(idx[i]) == found.end()) found.insert(idx[i]);
        else if (diagidx.find(idx[i]) == diagidx.end()) {
            diagidx.insert(std::map<size_t, size_t>::value_type(idx[i], d++));
        }
    }

    if (diagidx.size() == 0) {

        const symmetry<N, double> &sym = bto.get_symmetry();
        {
            gen_block_tensor_wr_ctrl<N, bti_traits> c(bt);
            so_copy<N, double>(sym).perform(c.req_symmetry());
        }
        ctf_btod_set<N>(val).perform(bt);

        if (n.add()) {
            addition_schedule<N, ctf_btod_traits> asch(sym, sym);
            {
                gen_block_tensor_rd_ctrl<N, bti_traits> cb(bt);
                std::vector<size_t> nzblk;
                cb.req_nonzero_blocks(nzblk);
                asch.build(bto.get_schedule(), nzblk);
            }

            scalar_transf<double> c(1.0);
            gen_bto_aux_add<N, ctf_btod_traits> out(sym, asch, bt, c);
            out.open();
            bto.perform(out);
            out.close();
        }
    }
    else {

        gen_bto_aux_copy<N, ctf_btod_traits> out(bto.get_symmetry(), bt);
        out.open();
        bto.perform(out);
        out.close();

        sequence<N, size_t> msk(0);
        for (size_t i = 0; i < N; i++) {
            std::map<size_t, size_t>::const_iterator j = diagidx.find(idx[i]);
            if (j == diagidx.end()) continue;

            msk[i] = j->second;
        }

        if (n.add()) ctf_btod_shift_diag<N>(msk, val).perform(bt);
        else ctf_btod_set_diag<N>(msk, val).perform(bt);
    }
}


} // unnamed namespace


template<size_t N>
set<N>::set(const expr_tree &tree, node_id_t id,
    const tensor_transf<N, double> &tr) :

    m_impl(new eval_set_impl<N>(tree, id, tr)) {

}


template<size_t N>
set<N>::~set() {

    delete m_impl;
}


template class set<1>;
template class set<2>;
template class set<3>;
template class set<4>;
template class set<5>;
template class set<6>;
template class set<7>;
template class set<8>;


} // namespace eval_ctf_btensor_double
} // namespace expr
} // namespace libtensor
