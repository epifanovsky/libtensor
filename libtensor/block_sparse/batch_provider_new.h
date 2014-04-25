#ifndef BATCH_PROVIDER_NEW_H
#define BATCH_PROVIDER_NEW_H

#include "batch_kernel.h"
#include "../core/scalar_transf_double.h"
#include "../expr/dag/node_assign.h"
#include "../expr/dag/node_transform.h"

namespace libtensor {

#if 0
namespace expr {

template<size_t M,typename T> 
class kernel_builder_2
{
    any_tensor<M,T> m_at;
    label<M> m_label;
public:
    kernel_builder_2(const node& n)
    {
        n.get_
    }
    template<size_t N,typename kernel_t>
    build(const kernel_builder_2<N>& rhs)
    {
        return kernel_t(m_at,rhs.get_any_tensor(),m_label,lhs.get_label());
    }
    }
    label<M>& get_any_tensor() { return m_label; } 
    any_tensor<M,T>& get_any_tensor() { return m_at; } 
}

template<typename T>
kernel_builder_2 build_kernel_builder_2(const node& n_one)
{
    switch(n_one.get_n())
    {
        case 1:
            return kernel_builder_2<1,T>(n_one);
        case 2:
            return kernel_builder_2<1>();
        case 3:
            return kernel_builder_2<1>();
        case 4:
            return kernel_builder_2<1>();
        case 5:
            return kernel_builder_2<1>();
        case 6:
            return kernel_builder_2<1>();
        case 7:
            return kernel_builder_2<1>();
        case 8:
            return kernel_builder_2<1>();
    }
}

template<typename kernel_t>
dispatch_two(const node& node_two)
{
    return p
}

} // namespace expr
#endif

template<typename T>
class batch_provider_new
{
private:
    batch_kernel<T>* m_kern;
public:
    static const char* k_clazz; //!< Class name
    batch_provider_new(const expr::expr_tree& tree); 
    void get_batch(T* output_ptr) {}
};

template<typename T>
const char* batch_provider_new<T>::k_clazz = "batch_provider<T>";

template<typename T>
batch_provider_new<T>::batch_provider_new(const expr::expr_tree& tree)
{
    using namespace expr;
    expr_tree::node_id_t root_id = tree.get_root();
    const node& root = tree.get_vertex(root_id);

    if(!root.check_type<node_assign>())
    {
        throw bad_parameter(g_ns,k_clazz,"batch_provider(...)",__FILE__, __LINE__,
            "Invalid root node");
    }

    const expr_tree::edge_list_t& children = tree.get_edges_out(root_id);
    if(children.size() != 2)
    {
        throw bad_parameter(g_ns,k_clazz,"batch_provider(...)",__FILE__, __LINE__,
            "Root node does not have right number of children");
    }

#if 0
    const node& kernel_node_id = children[1];
    const node& kernel_node = tree.get_vertex(kernel_node_id);
    const expr_tree::edge_list_t& inputs = tree.get_edges_out(kernel_node_id);
    if(kernel_node.check_type<node_transform>())
    {
        if(
        inputs.recast_as
        m_kernel = new batch_kernel_permute<T>(
    }
    else
    {
        throw bad_parameter(g_ns,k_clazz,"batch_provider(...)",__FILE__, __LINE__,
            "Invalid child node type");
    }
#endif
}


} // namespace libtensor

#endif /* BATCH_PROVIDER_NEW_H */
