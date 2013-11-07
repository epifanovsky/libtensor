#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_symmetrize2.h>
#include <libtensor/block_tensor/btod_symmetrize3.h>
#include <libtensor/expr/node_ident.h>
#include <libtensor/expr/node_symm.h>
#include "metaprog.h"
#include "node_inspector.h"
#include "eval_btensor_double_symm.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {

namespace {
using namespace libtensor::expr;


class eval_symm_impl {
private:
    enum {
        Nmax = symm::Nmax
    };

    typedef tensor_list::tid_t tid_t; //!< Tensor ID type

private:
    const tensor_list &m_tl; //!< Tensor list
    const interm &m_interm; //!< Intermediates
    const node_symm<double> &m_node; //!< Identity node
    bool m_add; //!< True if add

public:
    eval_symm_impl(const tensor_list &tl, const interm &inter, const node_symm<double> &node, bool add) :
        m_tl(tl), m_interm(inter), m_node(node), m_add(add)
    { }

    template<size_t N>
    void evaluate(const tensor_transf<N, double> &tr, tid_t tid);

    void evaluate(const tensor_transf<1, double> &tr, tid_t tid);

private:
    template<size_t N>
    btensor<N, double> &tensor_from_tid(tid_t tid);
    template<size_t N>
    btensor<N, double> &tensor_from_tid(tid_t tid,
        const block_index_space<N> &bis);

};


template<size_t N>
void eval_symm_impl::evaluate(const tensor_transf<N, double> &tr, tid_t tid) {

    if(m_node.get_nsym() == 2) {

        node_inspector ni(m_node.get_arg());
        node_with_transf<N> nwt = ni.gather_transf<N>();
        if(nwt.n.get_op().compare("ident") != 0) {
            throw "not identity node in symm";
        }

        // Need to convert
        // T2 S T1 A -> S' T' A, where S = I + Ts and S' = I + Ts'
        //
        // T2 (I + Ts) T1 A =
        // (T2 T1 + T2 Ts T1) A =
        // (T2 T1 + T2 Ts T2(inv) T2 T1) A =
        // [I + T2 Ts T2(inv)] T2 T1 A
        //
        // => Ts' = T2 Ts T2(inv); T' = T2 T1

        const node_ident &n1 = ni.extract_ident();
        btensor_i<N, double> &bta = tensor_from_tid<N>(n1.get_tid());

        if(N != m_tl.get_tensor_order(n1.get_tid())) {
            throw "Invalid order";
        }
        if(m_node.get_sym().size() % 2 != 0) {
            throw "Wrong size of symmetrization sequence";
        }
        size_t nsymidx = m_node.get_sym().size() / 2;
        permutation<N> symperm;
        for(size_t i = 0; i < nsymidx; i++) {
            symperm.permute(m_node.get_sym()[i], m_node.get_sym()[nsymidx + i]);
        }

        tensor_transf<N, double> tr2(tr), tr2inv(tr2, true);
        tensor_transf<N, double> tr1(nwt.tr);
        tensor_transf<N, double> tpr(tr1);
        tpr.transform(tr2);
        tensor_transf<N, double> trs(symperm, m_node.get_pair_tr());
        tensor_transf<N, double> tspr(tr2inv);
        tspr.transform(trs);
        tspr.transform(tr2);

        btod_copy<N> op(bta, tpr.get_perm(), tpr.get_scalar_tr().get_coeff());
        btod_symmetrize2<N> symop(op, tspr.get_perm(), tspr.get_scalar_tr().get_coeff() == 1.0);
        btensor<N, double> &bt = tensor_from_tid<N>(tid, symop.get_bis());
        if(m_add) {
            symop.perform(bt, 1.0);
        } else {
            symop.perform(bt);
        }

    } else if(m_node.get_nsym() == 3) {
        throw "Third-order symmetrizations not implemented";
    } else {
        throw "High-order symmetrizations not implemented";
    }
}


void eval_symm_impl::evaluate(const tensor_transf<1, double> &tr, tid_t tid) {

    throw "Should not be here";
}


template<size_t N>
btensor<N, double> &eval_symm_impl::tensor_from_tid(tid_t tid) {

    any_tensor<N, double> &anyt = m_tl.get_tensor<N, double>(tid);

    if(m_interm.is_interm(tid)) {
        btensor_placeholder<N, double> &ph =
            btensor_placeholder<N, double>::from_any_tensor(anyt);
        if(ph.is_empty()) throw 73;
        return ph.get_btensor();
    } else {
        return btensor<N, double>::from_any_tensor(anyt);
    }
}


template<size_t N>
btensor<N, double> &eval_symm_impl::tensor_from_tid(tid_t tid,
    const block_index_space<N> &bis) {

    any_tensor<N, double> &anyt = m_tl.get_tensor<N, double>(tid);

    if(m_interm.is_interm(tid)) {
        btensor_placeholder<N, double> &ph =
            btensor_placeholder<N, double>::from_any_tensor(anyt);
        if(ph.is_empty()) ph.create_btensor(bis);
        return ph.get_btensor();
    } else {
        return btensor<N, double>::from_any_tensor(anyt);
    }
}


} // unnamed namespace


template<size_t N>
void symm::evaluate(const tensor_transf<N, double> &tr, tid_t tid) {

    eval_symm_impl(m_tl, m_interm, m_node, m_add).evaluate(tr, tid);
}


//  The code here explicitly instantiates symm::evaluate<N>
namespace symm_ns {
template<size_t N>
struct aux {
    symm *e;
    tensor_transf<N, double> *tr;
    aux() { e->evaluate(*tr, 0); }
};
} // unnamed namespace
template class instantiate_template_1<1, symm::Nmax, symm_ns::aux>;


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
