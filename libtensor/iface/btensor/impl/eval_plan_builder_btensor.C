#include "metaprog.h"
#include "node_inspector.h"
#include "../eval_plan_builder_btensor.h"

namespace libtensor {
namespace iface {
using namespace expr;
using namespace eval_btensor_double;


namespace {

class node_renderer {
public:
    enum {
        Nmax = eval_plan_builder_btensor::Nmax
    };

    typedef eval_plan_builder_btensor::tid_t tid_t;

private:
    tensor_list &m_tl; //!< Tensor list
    const node &m_node; //!< Node
    tid_t m_tid; //!< Result tensor ID

public:
    node_renderer(tensor_list &tl, const node &n, tid_t tid) :
        m_tl(tl), m_node(n), m_tid(tid)
    { }

    void render() {
        dispatch_1<1, Nmax>::dispatch(*this, m_tl.get_tensor_order(m_tid));
    }

    template<size_t N>
    void dispatch() {

        node_inspector ni(m_node);
        node_with_transf<N> ntr = ni.gather_transf<N>();

        if(ntr.n.get_op().compare("ident") == 0) {

        }
    }

};

} // unnamed namespace


void eval_plan_builder_btensor::build_plan() {

    //  For now assume it's double
    //  TODO: implement other types

    tid_t tid = m_assign.get_tid();
    if(m_tl.get_tensor_type(tid) != typeid(double)) {
        throw "Bad tensor type";
    }

    node_renderer(m_tl, m_assign.get_rhs(), tid).render();
}


} // namespace iface
} // namespace libtensor
