#include <iostream>
#include <string>
#include <libtensor/expr/node_add.h>
#include <libtensor/expr/node_contract.h>
#include <libtensor/expr/print_node.h>
#include "metaprog.h"
#include "node_inspector.h"
#include "../eval_plan_builder_btensor.h"

namespace libtensor {
namespace iface {
using namespace expr;
using namespace eval_btensor_double;


const char eval_plan_builder_btensor::k_clazz[] = "eval_plan_builder_btensor";


namespace {

class node_renderer {
public:
    static const char k_clazz[];

public:
    enum {
        Nmax = eval_plan_builder_btensor::Nmax
    };

    typedef eval_plan_builder_btensor::tid_t tid_t;

private:
    eval_plan &m_plan; //!< Evaluation plan
    tensor_list &m_tl; //!< Tensor list
    const node &m_node; //!< Node
    tid_t m_tid; //!< Result tensor ID
    bool m_interm; //!< Whether the result is an intermediate
    bool m_asis; //!< Whether the node is to be used as is

public:
    node_renderer(eval_plan &plan, tensor_list &tl, const node &n, tid_t tid) :
        m_plan(plan), m_tl(tl), m_node(n), m_tid(tid), m_interm(false),
        m_asis(false)
    { }

    node_renderer(eval_plan &plan, tensor_list &tl, const node &n) :
        m_plan(plan), m_tl(tl), m_node(n), m_tid(0), m_interm(true),
        m_asis(false)
    { }

    void render() {
        dispatch_1<1, Nmax>::dispatch(*this, m_node.get_n());
    }

    tid_t get_tid() const {
        if(m_tid == 0) {
            static const char method[] = "get_tid()";
            throw not_implemented("iface", k_clazz, method, __FILE__, __LINE__);
        }
        return m_tid;
    }

    bool as_is() const {
        return m_asis;
    }

    template<size_t N>
    void dispatch() {

        static const char method[] = "dispatch()";

        if(m_node.get_op().compare("assign") == 0) {
            render_assign<N>();
        } else {
            node_inspector ni(m_node);
            node_with_transf<N> nwt = ni.gather_transf<N>();
            if(nwt.n.get_op().compare("add") == 0) {
                render_add(nwt);
            } else if(nwt.n.get_op().compare("contract") == 0) {
                render_contract(nwt);
            } else if(nwt.n.get_op().compare("ident") == 0) {
                render_ident(nwt);
            } else {
                throw not_implemented("iface", k_clazz, method, __FILE__, __LINE__);
            }
        }
    }

private:
    template<size_t N>
    void render_assign() {

        const node_assign &n = m_node.template recast_as<node_assign>();
        node_renderer r(m_plan, m_tl, n.get_rhs(), n.get_tid());
        r.render();
        if(r.as_is()) {
            print_node(n, std::cout);
            m_plan.insert_assignment(n);
        }
    }

    template<size_t N>
    void render_add(const node_with_transf<N> &nwt) {

        static const char method[] = "render_add()";

        const node_add &n = nwt.n.template recast_as<node_add>();

        if(m_interm) m_plan.create_intermediate(m_tid);
        std::vector<bool> visited(n.get_nargs(), false);

        for(size_t iarg = 0; iarg < visited.size(); iarg++) {
            node_inspector ni(n.get_arg(iarg));
            node_with_transf<N> nwt2 = ni.gather_transf<N>();
            if(nwt2.n.get_op().compare("ident") == 0) {
                add_assignment(node_with_transf<N>(n.get_arg(iarg), nwt.tr), true);
                visited[iarg] = true;
            }
        }

        for(size_t iarg = 0; iarg < visited.size(); iarg++) if(!visited[iarg]) {
            node_renderer r(m_plan, m_tl, n.get_arg(iarg), m_tid);
            r.render();
            if(r.as_is()) {
                add_assignment(node_with_transf<N>(n.get_arg(iarg),
                    tensor_transf<N, double>()), true);
            }
            visited[iarg] = true;
        }
    }

    template<size_t N>
    void render_contract(const node_with_transf<N> &nwt) {

        const node_contract &n = nwt.n.template recast_as<node_contract>();
        std::auto_ptr<node> a1, a2;

        node_renderer r1(m_plan, m_tl, n.get_arg(0));
        node_renderer r2(m_plan, m_tl, n.get_arg(1));
        r1.render();
        r2.render();

        if(r1.as_is() && r2.as_is()) {
            m_asis = true;
            return;
        }

        if(r1.as_is()) a1 = std::auto_ptr<node>(n.get_arg(0).clone());
        else a1 = std::auto_ptr<node>(new node_ident(r1.get_tid(), n.get_arg(0).get_n()));

        if(r2.as_is()) a2 = std::auto_ptr<node>(n.get_arg(1).clone());
        else a2 = std::auto_ptr<node>(new node_ident(r2.get_tid(), n.get_arg(1).get_n()));

        node_contract nc(*a1, *a2, n.get_contraction());
        if(m_interm) m_plan.create_intermediate(m_tid);
        add_assignment(node_with_transf<N>(nc, nwt.tr), false);
        if(!r1.as_is()) m_plan.delete_intermediate(r1.get_tid());
        if(!r2.as_is()) m_plan.delete_intermediate(r2.get_tid());
    }

    template<size_t N>
    void render_ident(const node_with_transf<N> &nwt) {

        m_asis = true;
    }

    template<size_t N>
    void add_assignment(const node_with_transf<N> &nwt, bool add) {

        if(nwt.tr.get_perm().is_identity() &&
            nwt.tr.get_scalar_tr().get_coeff() == 1.0) {

            node_assign na(m_tid, nwt.n, add);
            print_node(na, std::cout);
            m_plan.insert_assignment(na);

        } else {

            std::vector<size_t> perm(N);
            for(size_t i = 0; i < N; i++) perm[i] = nwt.tr.get_perm()[i];
            node_transform<double> ntr(nwt.n, perm, nwt.tr.get_scalar_tr());
            node_assign na(m_tid, ntr, add);
            print_node(na, std::cout);
            m_plan.insert_assignment(na);

        }
    }

};

const char node_renderer::k_clazz[] = "node_renderer";

} // unnamed namespace


void eval_plan_builder_btensor::build_plan() {

    static const char method[] = "build_plan()";

    //  For now assume it's double
    //  TODO: implement other types

    tid_t tid = m_assign.get_tid();
    if(m_tl.get_tensor_type(tid) != typeid(double)) {
        throw not_implemented("iface", k_clazz, method, __FILE__, __LINE__);
    }

    if(m_tl.get_tensor_order(tid) != m_assign.get_n()) {
        throw 111;
    }

    print_node(m_assign, std::cout);
    node_renderer(m_plan, m_tl, m_assign).render();
}


} // namespace iface
} // namespace libtensor
