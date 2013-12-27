#ifndef SPARSE_BLOCK_TREE_ITERATOR_H
#define SPARSE_BLOCK_TREE_ITERATOR_H

#include "sparse_block_tree_any_order.h"

//TODO REMOVE
#include <iostream>

namespace libtensor {

/* Used to allow for shared code between const and non-const iterators
 * A convenient workaround for the fact that member variables cannot be overwritten
 */
template<bool is_const>
class iterator_const_traits;

template<>
class iterator_const_traits<false> {
public:
    typedef sparse_block_tree_any_order* ptr_t;
    typedef sparse_block_tree_any_order::value_t& ref_t;
};

template<>
class iterator_const_traits<true> {
public:
    typedef const sparse_block_tree_any_order* ptr_t;
    typedef const sparse_block_tree_any_order::value_t& ref_t;
};

template<bool is_const>
class sparse_block_tree_iterator : iterator_const_traits<is_const>
{
private:
    typedef size_t key_t;
    typedef typename iterator_const_traits<is_const>::ptr_t ptr_t;
    typedef typename iterator_const_traits<is_const>::ref_t ref_t;

    std::vector<ptr_t> m_node_stack;
    std::vector<size_t> m_pos_stack;
public:
    ref_t operator*();
    std::vector<key_t> key();

    //Value of null is used to indicate end()
    sparse_block_tree_iterator(ptr_t root_node,const std::vector<size_t>& displacement = std::vector<size_t>());

    sparse_block_tree_iterator<is_const>& operator++();

    bool operator==(const sparse_block_tree_iterator<is_const>& rhs) const;
    bool operator!=(const sparse_block_tree_iterator<is_const>& rhs) const;
};

template<bool is_const>
sparse_block_tree_iterator<is_const>::sparse_block_tree_iterator(ptr_t root_node,const std::vector<size_t>& displacement)
{
    if(root_node != NULL)
    {
        std::vector<size_t> displacement_internal(displacement);
        if(displacement_internal.size() == 0)
        {
            for(size_t i = 0; i < root_node->m_order; ++i)
            {
                displacement_internal.push_back(0);
            }
        }

        m_node_stack.push_back(root_node);
        m_pos_stack.push_back(displacement_internal[0]);
        
        ptr_t cur_node = root_node;
        size_t m = 0;
        while(cur_node->m_order > 1)
        {
            cur_node = cur_node->m_children[displacement_internal[m]];
            ++m;
            m_node_stack.push_back(cur_node);
            m_pos_stack.push_back(displacement_internal[m]);
        }
    }
}

template<bool is_const>
typename sparse_block_tree_iterator<is_const>::ref_t sparse_block_tree_iterator<is_const>::operator*()
{
    return m_node_stack.back()->m_values[m_pos_stack.back()];
}

template<bool is_const>
std::vector<size_t> sparse_block_tree_iterator<is_const>::key()
{
    std::vector<key_t> the_key(m_node_stack.size());
    for(size_t i = 0; i < m_node_stack.size(); ++i)
    {
        the_key[i] = m_node_stack[i]->m_keys[m_pos_stack[i]];
    }
    return the_key; 
}

template<bool is_const>
sparse_block_tree_iterator<is_const>& sparse_block_tree_iterator<is_const>::operator++()
{
    //We try to increment starting from the deepest node 
    size_t rev_idx = 1;
    size_t order = m_node_stack.size();
    size_t cur_node_idx = order - rev_idx;
    while(rev_idx <= order)
    {
        size_t cur_pos = ++m_pos_stack[cur_node_idx];
        ptr_t cur_node = m_node_stack[cur_node_idx];

        //Exhausted this node? Move up a level
        if(cur_pos == cur_node->m_keys.size())
        {
            ++rev_idx;
        }
        else
        {
            //If we incremented a branch node, need to set lower levels appropriately
            if(cur_node->m_order > 1)
            {
                ptr_t lower_node = cur_node->m_children[cur_pos];
                for(size_t i = cur_node_idx+1; i < order; ++i)
                {
                    m_node_stack[i] = lower_node;
                    m_pos_stack[i] = 0;
                    if(i < order - 1)
                    {
                        lower_node = lower_node->m_children[0];
                    }
                } 
            }
            break;
        }
        cur_node_idx = order - rev_idx;
    }

    //Did we hit the end? If so, make us '==' end()
    if(rev_idx > order)
    {
        m_node_stack.clear();
    }
    return (*this);
}

template<bool is_const>
bool sparse_block_tree_iterator<is_const>::operator==(const sparse_block_tree_iterator<is_const>& rhs) const
{
    if(m_node_stack.size() != rhs.m_node_stack.size())
    {
        return false;
    }

    //end() == end()
    if(m_node_stack.size() == 0)
    {
        return true;
    }

    for(size_t i = 0; i < m_node_stack.size(); ++i)
    {
        if(m_node_stack[i] != rhs.m_node_stack[i])
        {
            return false;
        }
        else if(m_pos_stack[i] != rhs.m_pos_stack[i])
        {
            return false;
        }
    }
    return true;
}

template<bool is_const>
bool sparse_block_tree_iterator<is_const>::operator!=(const sparse_block_tree_iterator<is_const>& rhs) const
{
    return !(*this == rhs);
}

} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_ITERATOR_H */
