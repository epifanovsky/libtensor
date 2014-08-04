#ifndef SPARSE_BTENSOR_NEW_H
#define SPARSE_BTENSOR_NEW_H

#include <sstream>
#include <limits>
#include "sparse_bispace.h"
#include "sparse_loop_list.h"
#include "block_load_kernel.h"
#include "block_print_kernel.h"
#include "gen_sparse_btensor.h"
#include "batch_list_builder.h"
#include "memory_reserve.h"
#include "batch_provider.h"
#include <libtensor/expr/iface/expr_lhs.h>
#include <libtensor/expr/iface/labeled_lhs_rhs.h>
#include <libtensor/expr/operators/contract.h>
#include <libtensor/expr/operators/plus_minus.h>

namespace libtensor {

template<size_t N,typename T = double>
class sparse_btensor : public gen_sparse_btensor<N,T>,public expr::expr_lhs<N,T>
{
public:
    static const char *k_clazz; //!< Class name
private:
    T* m_data_ptr;
    sparse_bispace<N> m_bispace;
    memory_reserve* m_mr;
public:
    /** \brief Constructs a sparse block tensor object and populates it with the entries from mem if specified
     **/
    sparse_btensor(const sparse_bispace<N>& the_bispace,const T* mem = NULL,bool already_block_major = false);

    virtual ~sparse_btensor();

    //Copy constructor
    sparse_btensor(const sparse_btensor<N>& rhs);

    /** \brief Return the sparse_bispace defining this tensor 
     **/
    const sparse_bispace<N>& get_bispace() const; 

    /** \brief Compares the tensor to another
     *         Two sparse_btensors are equal if they have the same number of elements and all of those elements match
     **/
    bool operator==(const sparse_btensor<N,T>& rhs) const;
    bool operator!=(const sparse_btensor<N,T>& rhs) const;

    const T* get_data_ptr() const { return m_data_ptr; }
    virtual batch_provider_i<T>* get_batch_provider() const { return NULL; }
    
    /** \brief Returns a string representation of the tensor in row-major order 
     **/
    std::string str() const;

    void set_memory_reserve(memory_reserve& mr);

    virtual void assign(const expr::expr_rhs<N, T>& rhs, const expr::label<N>& l);

    expr::labeled_lhs_rhs<N, T> operator()(const expr::label<N> &label) {
        return expr::labeled_lhs_rhs<N, T>(*this, label,
            any_tensor<N, T>::make_rhs(label));
    }
};

template<size_t N,typename T>
const char *sparse_btensor<N,T>::k_clazz = "sparse_btensor<N,T>";

template<size_t N,typename T>
sparse_btensor<N,T>::sparse_btensor(const sparse_bispace<N>& the_bispace,const T* mem,bool already_block_major) : m_bispace(the_bispace),m_mr(NULL)
{
    //Determine size
    size_t size = 1;
    for(size_t i = 0; i < N; ++i)
    {
        size *= the_bispace[i].get_dim();
    }

    //Alloc storage
    m_data_ptr = new T[size];

    //Create loops

    if(mem != NULL)
    {
        if(already_block_major)
        {
            memcpy(m_data_ptr,mem,the_bispace.get_nnz()*sizeof(T));
        }
        else
        {

        	std::vector<sparse_bispace_any_order> bispaces(1,m_bispace);
            std::vector<block_loop> loops;
            for(size_t i = 0; i < N; ++i)
            {
            	block_loop loop(bispaces);
            	loop.set_subspace_looped(0,i);
            	loops.push_back(loop);
            }
        	sparse_loop_list sll(loops,bispaces);

            block_load_kernel<T> blk(m_bispace,mem);
            std::vector<T*> ptrs(1,m_data_ptr);
            sll.run(blk,ptrs);
        }
    }
    else
    {
        memset(m_data_ptr,0,size*sizeof(T));
    }
}

template<size_t N,typename T>
sparse_btensor<N,T>::~sparse_btensor()
{
    delete [] m_data_ptr;
    if(m_mr != NULL) m_mr->remove_tensor(this->m_bispace.get_nnz()*sizeof(T));
}

template<size_t N,typename T>
sparse_btensor<N,T>::sparse_btensor(const sparse_btensor<N>& rhs) : m_bispace(rhs.m_bispace),m_mr(NULL)
{
    this->set_memory_reserve(*rhs.m_mr);
    m_data_ptr = new T[m_bispace.get_nnz()]; 
    memcpy(m_data_ptr,rhs.m_data_ptr,m_bispace.get_nnz()*sizeof(T)); 
}

template<size_t N,typename T>
const sparse_bispace<N>& sparse_btensor<N,T>::get_bispace() const
{
    return m_bispace;
}



template<size_t N,typename T>
bool sparse_btensor<N,T>::operator==(const sparse_btensor<N,T>& rhs) const
{
    if(this->m_bispace.get_nnz() != rhs.m_bispace.get_nnz())
    {
        throw bad_parameter(g_ns, k_clazz,"operator==(...)",
                __FILE__, __LINE__, "tensors have different numbers of nonzero elements");
    }

    for(size_t i = 0; i < m_bispace.get_nnz(); ++i)
    {
        if(m_data_ptr[i] != rhs.m_data_ptr[i])
        {
            return false;
        }
    }
    return true;
}

template<size_t N,typename T>
bool sparse_btensor<N,T>::operator!=(const sparse_btensor<N,T>& rhs) const
{
    return !(*this == rhs);
}

template<size_t N,typename T>
std::string sparse_btensor<N,T>::str() const
{

    //Generate the loops for this tensor in slow->fast index order
	std::vector<sparse_bispace_any_order> bispaces(1,m_bispace);

    std::vector<block_loop> loops;
	for(size_t i = 0; i < N; ++i)
	{
		block_loop loop(bispaces);
		loop.set_subspace_looped(0,i);
		loops.push_back(loop);
	}
	sparse_loop_list sll(loops,bispaces);

	block_print_kernel<T> bpk;
	std::vector<T*> ptrs(1,m_data_ptr);
	sll.run(bpk,ptrs);
    return bpk.str();
}

template<size_t N,typename T>
void sparse_btensor<N,T>::set_memory_reserve(memory_reserve& mr)
{ 
    if(this->m_mr != NULL) this->m_mr->remove_tensor(this->m_bispace.get_nnz()*sizeof(T));
    this->m_mr = &mr;
    try
    {
        m_mr->add_tensor(this->m_bispace.get_nnz()*sizeof(T));
    }
    catch(out_of_memory&)
    {
        m_mr = NULL;
    }
}

template<size_t N,typename T>
void sparse_btensor<N,T>::assign(const expr::expr_rhs<N, T>& rhs, const expr::label<N>& l)
{
    using namespace expr;
    node_assign root(N);
    expr_tree e(root);
    expr_tree::node_id_t root_id = e.get_root();
    node_ident_any_tensor<N,T> n_tensor(*this);
    e.add(root_id,n_tensor);

    permutation<N> perm = l.permutation_of(rhs.get_label());
    if(!perm.is_identity()) 
    {
        std::vector<size_t> perm_entries(N);
        for(size_t i = 0; i < N; i++) perm_entries[i] = perm[i];

        node_transform<T> n_tf(perm_entries, scalar_transf<T>());
        root_id = e.add(root_id,n_tf);
    }
    e.add(root_id, rhs.get_expr());

    //We clean up the tree a little bit to prevent spurious extra assignments
    std::vector<expr_tree::node_id_t> node_stack(1,e.get_root());
    std::vector<size_t> pos_stack(1,0);
    while(node_stack.size() > 0)
    {
        expr_tree::node_id_t cur_id = node_stack.back();
        const node& n = e.get_vertex(cur_id);
        if(n.check_type<node_assign>())
        {
            expr_tree::edge_list_t children = e.get_edges_out(cur_id);
            expr_tree::node_id_t op_id = children[1];
            const node& n_op = e.get_vertex(op_id); 
            if(n_op.check_type<node_assign>())
            {
                expr_tree::edge_list_t op_children = e.get_edges_out(op_id);
                expr_tree sub = e.get_subtree(op_children[1]);
                e.erase_subtree(op_id);
                e.add(cur_id,sub);
            }
        }
        expr_tree::edge_list_t children = e.get_edges_out(cur_id);
        if(children.size() > 0 && pos_stack.back() < children.size())
        {
            node_stack.push_back(children[pos_stack.back()]);
            pos_stack.push_back(0);
        }
        else
        {
            node_stack.pop_back();
            pos_stack.pop_back();
            if(pos_stack.size() > 0) pos_stack.back()++;
        }
    }
    batch_provider<T> bp(e);

    std::vector< std::vector<sparse_bispace_any_order> > direct_bispace_grps;
    std::vector<idx_list> batched_subspace_grps;
    bp.get_direct_bispace_grps(direct_bispace_grps);
    bp.get_batched_subspace_grps(batched_subspace_grps);
    batch_list_builder blb(direct_bispace_grps,batched_subspace_grps);
    size_t mem_avail = (m_mr != NULL) ? m_mr->get_mem_avail() : std::numeric_limits<double>::max();
    idx_pair_list batch_list = blb.get_batch_list(mem_avail/sizeof(T));
    bp.set_batch_info(batched_subspace_grps,batch_list);

    bp.get_batch(this->m_data_ptr);
}

} // namespace libtensor

#endif /* SPARSE_btensor_H */
