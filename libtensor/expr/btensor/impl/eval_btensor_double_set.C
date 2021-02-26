#include <memory>
#include <vector>
#include <libtensor/block_tensor/bto_copy.h>
#ifdef WITH_LIBXM
#include <libtensor/block_tensor/bto_copy_xm.h>
#endif // WITH_LIBXM
#include <libtensor/block_tensor/bto_set.h>
#include <libtensor/block_tensor/bto_set_diag.h>
#include <libtensor/block_tensor/bto_shift_diag.h>
#include <libtensor/expr/dag/node_const_scalar.h>
#include <libtensor/expr/dag/node_set.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/eval/eval_exception.h>
#include <libtensor/gen_block_tensor/addition_schedule.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/symmetry/so_copy.h>
#include "tensor_from_node.h"
#include "eval_btensor_double_set.h"
#include "eval_btensor_double_autoselect.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

extern bool use_libxm;

namespace {


template<size_t N, typename T>
class eval_set_impl : public eval_btensor_evaluator_i<N, T> {
private:
    enum {
        Nmax = set<N, T>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<N, T>::bti_traits bti_traits;

private:
    btensor_i<N, T> *m_bt; //!< Tensor
    additive_gen_bto<N, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_set_impl(const expr_tree &tree, expr_tree::node_id_t id,
        const tensor_transf<N, T> &tr);

    virtual ~eval_set_impl();

    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return *m_op;
    }

private:
    static void perform_op(
        const node_set &n,
        additive_gen_bto<N, bti_traits> &bto,
        T v,
        btensor<N, T> &bt);
};

/*
template<size_t N>
additive_gen_bto<N, typename eval_set_impl<N, float>::bti_traits> *create_op(
    btensor_i<N, float> &bt, const tensor_transf<N, float> &tr,
    bool use_libxm) {

    return new bto_copy<N, float>(bt, tr.get_perm(),
        tr.get_scalar_tr().get_coeff());
}
*/

template<size_t N, typename T>
additive_gen_bto<N, typename eval_set_impl<N, T>::bti_traits> *create_op(
    btensor_i<N, T> &bt, const tensor_transf<N, T> &tr,
    bool use_libxm) {

#ifdef WITH_LIBXM
    if(use_libxm) {
        return new bto_copy_xm<N, T>(bt, tr.get_perm(),
            tr.get_scalar_tr().get_coeff());
    } else {
        return new bto_copy<N, T>(bt, tr.get_perm(),
            tr.get_scalar_tr().get_coeff());
    }
#else // WITH_LIBXM
    return new bto_copy<N, T>(bt, tr.get_perm(),
        tr.get_scalar_tr().get_coeff());
#endif // WITH_LIBXM
}


template<size_t N, typename T>
eval_set_impl<N, T>::eval_set_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<N, T> &tr) {

    const node_set &n = tree.get_vertex(id).template recast_as<node_set>();

    // Retrieve tensor argument and respective tensor operation
    const expr_tree::edge_list_t &e = tree.get_edges_out(id);
    tensor_transf<N, T> trx;
    expr_tree::node_id_t rhs = transf_from_node(tree, e[0], trx);

    autoselect<N, T> eval(tree, rhs, trx);
    additive_gen_bto<N, bti_traits> &op = eval.get_bto();

    // Retrieve scalar argument
    const node_const_scalar<T> &ns = tree.get_vertex(e[1]).
            template recast_as< node_const_scalar<T> >();
    const T &val = ns.get_scalar();

    // Create tensor
    std::unique_ptr< btensor<N, T> > bt(new btensor<N, T>(op.get_bis()));
    perform_op(n, op, val, *bt);

    m_bt = bt.release();
    m_op = create_op(*m_bt, tr, use_libxm);
}


template<size_t N, typename T>
eval_set_impl<N, T>::~eval_set_impl() {

    delete m_op;
    delete m_bt;
}


template<size_t N, typename T>
void eval_set_impl<N, T>::perform_op(
    const node_set &n,
    additive_gen_bto<N, bti_traits> &bto,
    T val,
    btensor<N, T> &bt) {

    const std::vector<size_t> &idx = n.get_idx();
    if (idx.size() != N) {
        throw eval_exception(__FILE__, __LINE__,
                "libtensor::expr::eval_btensor_T", "eval_set_impl<N>",
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

        const symmetry<N, T> &sym = bto.get_symmetry();
        {
            block_tensor_wr_ctrl<N, T> c(bt);
            so_copy<N, T>(sym).perform(c.req_symmetry());
        }
        bto_set<N, T>(val).perform(bt);

        if (n.add()) {
            addition_schedule<N, bto_traits<T> > asch(sym, sym);
            {
                gen_block_tensor_rd_ctrl<N, bti_traits> cb(bt);
                std::vector<size_t> nzblk;
                cb.req_nonzero_blocks(nzblk);
                asch.build(bto.get_schedule(), nzblk);
            }

            scalar_transf<T> c(1.0);
            gen_bto_aux_add<N, bto_traits<T> > out(sym, asch, bt, c);
            out.open();
            bto.perform(out);
            out.close();
        }
    }
    else {

        gen_bto_aux_copy<N, bto_traits<T> > out(bto.get_symmetry(), bt);
        out.open();
        bto.perform(out);
        out.close();

        sequence<N, size_t> msk(0);
        for (size_t i = 0; i < N; i++) {
            std::map<size_t, size_t>::const_iterator j = diagidx.find(idx[i]);
            if (j == diagidx.end()) continue;

            msk[i] = j->second;
        }

        if (n.add()) bto_shift_diag<N, T>(msk, val).perform(bt);
        else bto_set_diag<N, T>(msk, val).perform(bt);
    }
}


} // unnamed namespace


template<size_t N, typename T>
set<N, T>::set(const expr_tree &tree, node_id_t id,
    const tensor_transf<N, T> &tr) :

    m_impl(new eval_set_impl<N, T>(tree, id, tr)) {

}


template<size_t N, typename T>
set<N, T>::~set() {

    delete m_impl;
}


template class set<1, double>;
template class set<2, double>;
template class set<3, double>;
template class set<4, double>;
template class set<5, double>;
template class set<6, double>;
template class set<7, double>;
template class set<8, double>;

template class set<1, float>;
template class set<2, float>;
template class set<3, float>;
template class set<4, float>;
template class set<5, float>;
template class set<6, float>;
template class set<7, float>;
template class set<8, float>;

} // namespace eval_btensor_T
} // namespace expr
} // namespace libtensor
