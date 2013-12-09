#ifndef SPARSE_BLOCK_TREE_ANY_ORDER_NEW_H
#define SPARSE_BLOCK_TREE_ANY_ORDER_NEW_H

#include <vector>
#include "../core/sequence.h"

namespace libtensor { 

namespace impl {

template<typename key_t,typename value_t,bool is_const>
class sparse_block_tree_iterator_new;

template<typename key_t,typename value_t>
class sparse_block_tree_node_new
{
private:
    size_t m_order;
    std::vector<key_t> m_keys;
    //For branch nodes only
    std::vector< sparse_block_tree_node_new* > m_children;
    //For leaf nodes only
    std::vector<value_t> m_values;

    //Used by primary constructor to add new entries 
    template<size_t N>
    void push_back(const sequence<N,key_t>& key);

    //Used by push_back to create new branch nodes below the root
    template<size_t N>
    sparse_block_tree_node_new(const sequence<N,key_t>& key,size_t order);

protected:
    //We don't want these to be directly instantiable - force people to use the templated interface instead
    //This is called by the order-templated subclass
    template<size_t N>
    sparse_block_tree_node_new(const std::vector< sequence<N,key_t> >& sig_blocks);
public:
    typedef sparse_block_tree_iterator_new<key_t,value_t,false> iterator;
    bool operator==(const sparse_block_tree_node_new<key_t,value_t>& rhs) const;
    bool operator!=(const sparse_block_tree_node_new<key_t,value_t>& rhs) const;

    iterator begin();
    iterator end();

    template<typename other_key_t,typename other_value_t,bool is_const>
    friend class sparse_block_tree_iterator_new;
};


template<typename key_t,typename value_t> template<size_t N>
sparse_block_tree_node_new<key_t,value_t>::sparse_block_tree_node_new(const sequence<N,key_t>& key,size_t order)
{
    m_order = order;
    push_back(key);
}

template<typename key_t,typename value_t> template<size_t N>
void sparse_block_tree_node_new<key_t,value_t>::push_back(const sequence<N,key_t>& key)
{
    m_keys.push_back(key[N - m_order]);
    if(m_order > 1)
    {
        m_children.push_back(new sparse_block_tree_node_new(key,m_order - 1));
    }
    else
    {
        m_values.push_back(value_t());
    }
}

template<typename key_t,typename value_t> template<size_t N>
sparse_block_tree_node_new<key_t,value_t>::sparse_block_tree_node_new(const std::vector< sequence<N,key_t> >& sig_blocks)
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

template<typename key_t,typename value_t>
bool sparse_block_tree_node_new<key_t,value_t>::operator==(const sparse_block_tree_node_new<key_t,value_t>& rhs) const
{
    for(size_t i = 0; i < m_keys.size(); ++i)
    {
        if(m_keys[i] != rhs.m_keys[i])
        {
            return false;
        }
        if(m_order == 1)
        {
            if(m_values[i] != rhs.m_values[i])
            {
                return false;
            }
        }
        else
        {
            if((*m_children[i]) != (*rhs.m_children[i]))
            {
                return false;
            }
        }
    }
    return true;
}

template<typename key_t,typename value_t>
bool sparse_block_tree_node_new<key_t,value_t>::operator!=(const sparse_block_tree_node_new<key_t,value_t>& rhs) const
{
    return !(*this == rhs);
}

/* Used to allow for shared code between const and non-const iterators
 * A convenient workaround for the fact that member variables cannot be overwritten
 */
template<typename key_t,typename value_t,bool is_const>
class iterator_const_traits_new;

template<typename key_t,typename value_t>
class iterator_const_traits_new<key_t,value_t,false> {
public:
    typedef sparse_block_tree_node_new<key_t,value_t>* ptr_t;
};

template<typename key_t,typename value_t>
class iterator_const_traits_new<key_t,value_t,true> {
public:
    typedef const sparse_block_tree_node_new<key_t,value_t>* ptr_t;
};

template<typename key_t,typename value_t,bool is_const>
class sparse_block_tree_iterator_new : iterator_const_traits_new<key_t,value_t,is_const>
{
private:
    typedef typename iterator_const_traits_new<key_t,value_t,is_const>::ptr_t ptr_t;
    std::vector<ptr_t> m_node_stack;
    std::vector<size_t> m_pos_stack;
public:
    value_t& operator*();
    std::vector<key_t> key();

    //Value of null is used to indicate end()
    sparse_block_tree_iterator_new(ptr_t root_node);

    sparse_block_tree_iterator_new<key_t,value_t,is_const>& operator++();

    bool operator==(const sparse_block_tree_iterator_new<key_t,value_t,is_const>& rhs) const;
    bool operator!=(const sparse_block_tree_iterator_new<key_t,value_t,is_const>& rhs) const;
};

template<typename key_t,typename value_t,bool is_const>
sparse_block_tree_iterator_new<key_t,value_t,is_const>::sparse_block_tree_iterator_new(ptr_t root_node)
{
    if(root_node != NULL)
    {
        m_node_stack.push_back(root_node);
        m_pos_stack.push_back(0);
        
        ptr_t cur_node = root_node;
        while(cur_node->m_order > 1)
        {
            cur_node = cur_node->m_children[0];
            m_node_stack.push_back(cur_node);
            m_pos_stack.push_back(0);
        }
    }
}

template<typename key_t,typename value_t,bool is_const>
value_t& sparse_block_tree_iterator_new<key_t,value_t,is_const>::operator*()
{
    return m_node_stack.back()->m_values[m_pos_stack.back()];
}

template<typename key_t,typename value_t,bool is_const>
std::vector<key_t> sparse_block_tree_iterator_new<key_t,value_t,is_const>::key()
{
    std::vector<key_t> key(m_node_stack.size());
    for(size_t i = 0; i < m_node_stack.size(); ++i)
    {
        key[i] = m_node_stack[i]->m_keys[m_pos_stack[i]];
    }
    return key; 
}

template<typename key_t,typename value_t,bool is_const>
sparse_block_tree_iterator_new<key_t,value_t,is_const>& sparse_block_tree_iterator_new<key_t,value_t,is_const>::operator++()
{
    //We try to increment starting from the deepest node 
    size_t rev_idx = 1;
    size_t order = m_node_stack.size();
    size_t cur_node_idx = order - rev_idx;
    while(cur_node_idx <= order)
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

template<typename key_t,typename value_t,bool is_const>
bool sparse_block_tree_iterator_new<key_t,value_t,is_const>::operator==(const sparse_block_tree_iterator_new<key_t,value_t,is_const>& rhs) const
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

template<typename key_t,typename value_t,bool is_const>
bool sparse_block_tree_iterator_new<key_t,value_t,is_const>::operator!=(const sparse_block_tree_iterator_new<key_t,value_t,is_const>& rhs) const
{
    return !(*this == rhs);
}

template<typename key_t,typename value_t>
sparse_block_tree_iterator_new<key_t,value_t,false> sparse_block_tree_node_new<key_t,value_t>::begin()
{
    if(m_keys.size() == 0)
    {
        return iterator(NULL);
    }
    else
    {
        return iterator(this);
    }
}

template<typename key_t,typename value_t>
sparse_block_tree_iterator_new<key_t,value_t,false> sparse_block_tree_node_new<key_t,value_t>::end()
{
    return iterator(NULL);
}

} // namespace impl

} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_ANY_ORDER_H */
