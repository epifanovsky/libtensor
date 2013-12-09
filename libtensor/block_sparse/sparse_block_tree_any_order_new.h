#ifndef SPARSE_BLOCK_TREE_ANY_ORDER_NEW_H
#define SPARSE_BLOCK_TREE_ANY_ORDER_NEW_H

#include <vector>
#include "../core/sequence.h"

namespace libtensor { 

namespace impl {

template<bool is_const>
class sparse_block_tree_iterator_new;

class sparse_block_tree_any_order_new
{
private:
    typedef size_t key_t;
    typedef std::vector<size_t> value_t;

    size_t m_order;
    std::vector<key_t> m_keys;
    //For branch nodes only
    std::vector< sparse_block_tree_any_order_new* > m_children;
    //For leaf nodes only
    std::vector<value_t> m_values;

    //Used by primary constructor to add new entries 
    template<size_t N>
    void push_back(const sequence<N,key_t>& key);

    //Used by push_back to create new branch nodes below the root
    template<size_t N>
    sparse_block_tree_any_order_new(const sequence<N,key_t>& key,size_t order);

protected:
    //We don't want these to be directly instantiable - force people to use the templated interface instead
    //This is called by the order-templated subclass
    template<size_t N>
    sparse_block_tree_any_order_new(const std::vector< sequence<N,key_t> >& sig_blocks);
public:
    typedef sparse_block_tree_iterator_new<false> iterator;
    bool operator==(const sparse_block_tree_any_order_new& rhs) const;
    bool operator!=(const sparse_block_tree_any_order_new& rhs) const;

    iterator begin();
    iterator end();

    template<bool is_const>
    friend class sparse_block_tree_iterator_new;
};

template<size_t N>
sparse_block_tree_any_order_new::sparse_block_tree_any_order_new(const sequence<N,size_t>& key,size_t order)
{
    m_order = order;
    push_back(key);
}

template<size_t N>
void sparse_block_tree_any_order_new::push_back(const sequence<N,size_t>& key)
{
    m_keys.push_back(key[N - m_order]);
    if(m_order > 1)
    {
        m_children.push_back(new sparse_block_tree_any_order_new(key,m_order - 1));
    }
    else
    {
        m_values.push_back(value_t());
    }
}

template<size_t N>
sparse_block_tree_any_order_new::sparse_block_tree_any_order_new(const std::vector< sequence<N,size_t> >& sig_blocks)
{
    if(N == 0)
    {
        throw bad_parameter(g_ns,"sparse_block_tree_node","sparse_block_tree_node(...)",
                            __FILE__,__LINE__,"cannot pass 0 dimensional sequence as argument");
    }
    m_order = N;

    if(sig_blocks.size() != 0)
    {
        push_back(sig_blocks[0]);
    }
    
    //Ensure that block list is sorted in lexicographic order
    for(size_t i = 1; i < sig_blocks.size(); ++i)
    {
        const sequence<N,key_t>& cur = sig_blocks[i];
        const sequence<N,key_t>& prev = sig_blocks[i-1];

        bool equal = true;
        for(size_t j = 0; j < m_order; ++j)
        {
            if(cur[j] < prev[j])
            {
                throw bad_parameter(g_ns,"sparse_block_tree_node","sparse_block_tree_node(...)",
                    __FILE__,__LINE__,"list is not strictly increasing"); 
            }
            else if(cur[j] > prev[j])
            {
                equal = false;
                break;
            }
        }

        if(equal)
        {
            throw bad_parameter(g_ns,"sparse_block_tree<N>","sparse_block_tree(...)",
                __FILE__,__LINE__,"duplicate keys are not allowed"); 
        }
        push_back(cur);
    }
}

} // namespace impl

} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_ANY_ORDER_H */
