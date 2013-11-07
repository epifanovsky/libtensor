#include <iostream>
#include <string>
#include <libtensor/expr/node_add.h>
#include <libtensor/expr/node_contract.h>
#include <libtensor/expr/node_symm.h>
#include <libtensor/expr/print_node.h>
#include "btensor_placeholder.h"
#include "metaprog.h"
#include "node_inspector.h"
#include "eval_plan_builder_btensor.h"

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
    interm &m_interm; //!< Intermediates container
    const node &m_node; //!< Node
    tid_t m_out_tid; //!< Result tensor ID
    bool m_out_interm; //!< Whether the result is an intermediate
    bool m_out_asis; //!< Whether the node is to be used as is

public:
    node_renderer(eval_plan &plan, tensor_list &tl, interm &inter,
        const node &n, tid_t tid) :

        m_plan(plan), m_tl(tl), m_interm(inter), m_node(n), m_out_tid(tid),
        m_out_interm(false), m_out_asis(false)
    { }

    node_renderer(eval_plan &plan, tensor_list &tl, interm &inter,
        const node &n) :

        m_plan(plan), m_tl(tl), m_interm(inter), m_node(n), m_out_tid(0),
        m_out_interm(true), m_out_asis(false)
    { }

    void render() {
        dispatch_1<1, Nmax>::dispatch(*this, m_node.get_n());
    }

    tid_t get_tid() const {

        if(m_out_tid == 0) {
            static const char method[] = "get_tid()";
            throw not_implemented("iface", k_clazz, method, __FILE__, __LINE__);
        }
        return m_out_tid;
    }

    bool as_is() const {
        return m_out_asis;
    }

    template<size_t N>
    void dispatch() {

        if(m_node.get_op().compare("assign") == 0) {
            render_assign<N>();
        } else {
            render(tensor_transf<N, double>());
        }
    }

private:
    template<size_t N>
    void render(const tensor_transf<N, double> &tr) {

        node_inspector ni(m_node);
        node_with_transf<N> nwt = ni.gather_transf<N>();
        tensor_transf<N, double> tr1(nwt.tr);
        tr1.transform(tr);
        render(nwt.n, tr1);
    }

    template<size_t N>
    void render(const node &n, const tensor_transf<N, double> &tr) {

        if(n.get_op().compare("add") == 0) {
            render_add(n.template recast_as<node_add>(), tr);
        } else if(n.get_op().compare("contract") == 0) {
            render_contract(n.template recast_as<node_contract>(), tr);
        } else if(n.get_op().compare("ident") == 0) {
            render_ident<N>();
        } else if(n.get_op().compare("symm") == 0) {
            render_symm(n.template recast_as< node_symm<double> >(), tr);
        } else {
            throw not_implemented("iface", k_clazz, "render", __FILE__, __LINE__);
        }
    }

    template<size_t N>
    void render_assign() {

        const node_assign &n = m_node.template recast_as<node_assign>();
        node_renderer r(m_plan, m_tl, m_interm, n.get_rhs(), n.get_tid());
        r.render();
        if(r.as_is()) {
            print_node(n, std::cout);
            m_plan.insert_assignment(n);
        }
    }

    template<size_t N>
    void render_add(const node_add &n, const tensor_transf<N, double> &tr) {

        static const char method[] = "render_add()";

        if(m_out_interm) {
            m_out_tid = m_interm.create_interm<N, double>();
            m_plan.create_intermediate(m_out_tid);
        }

        std::vector<bool> visited(n.get_nargs(), false);

        for(size_t iarg = 0; iarg < visited.size(); iarg++) {
            node_inspector ni(n.get_arg(iarg));
            node_with_transf<N> nwt2 = ni.gather_transf<N>();
            if(nwt2.n.get_op().compare("ident") == 0) {
                add_assignment(node_with_transf<N>(n.get_arg(iarg), tr), true);
                visited[iarg] = true;
            }
        }

        for(size_t iarg = 0; iarg < visited.size(); iarg++) if(!visited[iarg]) {
            node_renderer r(m_plan, m_tl, m_interm, n.get_arg(iarg), m_out_tid);
            r.render(tr);
            if(r.as_is()) {
                add_assignment(node_with_transf<N>(n.get_arg(iarg), tr), true);
            }
            visited[iarg] = true;
        }
    }

    template<size_t N>
    void render_contract(const node_contract &n,
        const tensor_transf<N, double> &tr) {

        std::auto_ptr<node> a1, a2;

        node_renderer r1(m_plan, m_tl, m_interm, n.get_arg(0));
        node_renderer r2(m_plan, m_tl, m_interm, n.get_arg(1));
        r1.render();
        r2.render();

        if(r1.as_is()) a1 = std::auto_ptr<node>(n.get_arg(0).clone());
        else a1 = std::auto_ptr<node>(new node_ident(r1.get_tid(), n.get_arg(0).get_n()));

        if(r2.as_is()) a2 = std::auto_ptr<node>(n.get_arg(1).clone());
        else a2 = std::auto_ptr<node>(new node_ident(r2.get_tid(), n.get_arg(1).get_n()));

        if(m_out_interm) {
            m_out_tid = m_interm.create_interm<N, double>();
            m_plan.create_intermediate(m_out_tid);
        }

        node_contract nc(*a1, *a2, n.get_contraction());
        add_assignment(node_with_transf<N>(nc, tr), true);
        if(!r1.as_is()) m_plan.delete_intermediate(r1.get_tid());
        if(!r2.as_is()) m_plan.delete_intermediate(r2.get_tid());
    }

    template<size_t N>
    void render_ident() {

        m_out_asis = true;
    }

    template<size_t N>
    void render_symm(const node_symm<double> &n,
        const tensor_transf<N, double> &tr) {

        std::auto_ptr<node> a1;

        node_renderer r1(m_plan, m_tl, m_interm, n.get_arg());
        r1.render();

        if(r1.as_is()) a1 = std::auto_ptr<node>(n.get_arg().clone());
        else a1 = std::auto_ptr<node>(new node_ident(r1.get_tid(), n.get_arg().get_n()));

        if(m_out_interm) {
            m_out_tid = m_interm.create_interm<N, double>();
            m_plan.create_intermediate(m_out_tid);
        }

        node_symm<double> ns(*a1, n.get_sym(), n.get_nsym(), n.get_pair_tr(), n.get_cyclic_tr());
        add_assignment(node_with_transf<N>(ns, tr), true);
        if(!r1.as_is()) m_plan.delete_intermediate(r1.get_tid());
    }

    template<size_t N>
    void add_assignment(const node_with_transf<N> &nwt, bool add) {

        std::cout << "add node to plan " << (void*)m_out_tid << std::endl;
        if(nwt.tr.get_perm().is_identity() &&
            nwt.tr.get_scalar_tr().get_coeff() == 1.0) {

            node_assign na(m_out_tid, nwt.n, add);
            print_node(na, std::cout);
            m_plan.insert_assignment(na);

        } else {

            std::vector<size_t> perm(N);
            for(size_t i = 0; i < N; i++) perm[i] = nwt.tr.get_perm()[i];
            node_transform<double> ntr(nwt.n, perm, nwt.tr.get_scalar_tr());
            node_assign na(m_out_tid, ntr, add);
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

    std::cout << "render expression" << std::endl;
    print_node(m_assign, std::cout);
    node_renderer(m_plan, m_tl, m_interm, m_assign).render();
}


} // namespace iface
} // namespace libtensor
