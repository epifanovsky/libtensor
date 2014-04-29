#ifndef BATCH_PROVIDER_NEW_H
#define BATCH_PROVIDER_NEW_H

#include "gen_sparse_btensor.h"
#include "batch_kernel_permute.h"
#include "../expr/dag/node_assign.h"
#include "../expr/dag/node_transform.h"
#include "../expr/dag/node_contract.h"

namespace libtensor {

namespace expr {

template<typename T>
batch_kernel<T>* dispatch_one(const node& n_one,const node& n_two,const node& op_node)
{
    switch(n_one.get_n())
    {
        case 3:
            const node_ident_any_tensor<3,T>& n_at = dynamic_cast< const node_ident_any_tensor<3,T>& >(n_one);
            return dispatch_two(dynamic_cast< const gen_sparse_btensor<3,T>& >(n_at.get_tensor()),n_two,op_node);
    }
}

template<size_t M,typename T>
batch_kernel<T>* dispatch_two(const gen_sparse_btensor<M,T>& input_one,const node& n_two,const node& op_node)
{
    switch(n_two.get_n())
    {
        case 3:
            const node_ident_any_tensor<3,T>& n_at = dynamic_cast< const node_ident_any_tensor<3,T>& >(n_two);
            return dispatch_op(input_one,dynamic_cast< const gen_sparse_btensor<3,T>& >(n_at.get_tensor()),op_node);
    }
}

template<size_t M,size_t N,typename T>
batch_kernel<T>* dispatch_op(const gen_sparse_btensor<M,T>& input_one,const gen_sparse_btensor<N,T>& input_two,const node& op_node)
{
    if(op_node.check_type<node_transform_base>())
    {
        const node_transform_base& n_tf = dynamic_cast< const node_transform_base& >(op_node);
        return new batch_kernel_permute<T>(input_one,input_two,n_tf.get_perm());
    }
    else
    {
        throw bad_parameter(g_ns,"dispatch_op","",__FILE__, __LINE__,
            "Unsupported op node type");
    }
}

template<typename T>
T* get_data_ptr_from_node(const node& n_one)
{
    switch(n_one.get_n())
    {
        case 3:
            const node_ident_any_tensor<3,T>& n_at = dynamic_cast< const node_ident_any_tensor<3,T>& >(n_one);
            return (T*)static_cast< gen_sparse_btensor<3,T>& >(n_at.get_tensor()).get_data_ptr();
    }
}

} // namespace expr

template<typename T>
class batch_provider_new
{
private:
    batch_kernel<T>* m_kern;
    std::vector<T*> m_ptrs;
public:
    static const char* k_clazz; //!< Class name
    batch_provider_new(const expr::expr_tree& tree); 
    void get_batch(T* output_ptr); 
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

    const node& out_node = tree.get_vertex(children[0]);
    const node& op_node = tree.get_vertex(children[1]);
    const expr_tree::edge_list_t& inputs = tree.get_edges_out(children[1]);
    if(op_node.check_type<node_transform_base>())
    {
        const node& input_node = tree.get_vertex(inputs[0]);
        m_kern = dispatch_one<T>(out_node,input_node,op_node);
        m_ptrs.resize(2);
        m_ptrs[1] = get_data_ptr_from_node<T>(input_node);
    }
#if 0
    else if(op_node.check_type<node_contract>()) 
    {
        const node& A_node = tree.get_vertex(inputs[0]);
        const node& B_node = tree.get_vertex(inputs[1]);
        m_kern = dispatch_one<T>(out_node,input_node,op_node);
        m_ptrs.resize(3);
        m_ptrs[1] = get_data_ptr_from_node<T>(A_node);
        m_ptrs[2] = get_data_ptr_from_node<T>(B_node);
    }
#endif
    else
    {
        throw bad_parameter(g_ns,k_clazz,"batch_provider(...)",__FILE__, __LINE__,
            "Invalid child node type");
    }
}

template<typename T>
void batch_provider_new<T>::get_batch(T* output_ptr)
{ 
    m_ptrs[0] = output_ptr; 
    m_kern->generate_batch(m_ptrs,bispace_batch_map()); 
}


} // namespace libtensor

#endif /* BATCH_PROVIDER_NEW_H */
