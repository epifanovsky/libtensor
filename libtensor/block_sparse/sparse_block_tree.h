#ifndef SPARSE_BLOCK_TREE_H
#define SPARSE_BLOCK_TREE_H

#include <vector>
#include <algorithm>
#include <utility>
#include "../core/sequence.h"

//TODO REMOVE
#include <iostream>

namespace libtensor {

namespace impl
{

//Abstract node interface for all nodes
class sparse_block_tree_node_i {
protected:
    std::vector<size_t> m_keys;
public:
    typedef std::vector<size_t>::iterator sub_key_iterator;
    virtual sub_key_iterator get_sub_key_iterator(const std::vector<size_t>& sub_key,size_t cur_idx) = 0;
};

//Forward declaration for friendship
template<size_t N>
class sparse_block_tree_iterator;

//Forward declaration to allow specialization
template<size_t N>
class sparse_block_tree_node;

//"Leaf" node, can store values as well as keys
template<>
class sparse_block_tree_node<1> : public sparse_block_tree_node_i {
protected:
    std::vector<size_t> m_values;
public:
    //DO NOT CALL...only for std::vector compatibility
    sparse_block_tree_node() {}; 

    //Base case for the recursion
    sub_key_iterator get_sub_key_iterator(const std::vector<size_t>& sub_key,size_t cur_idx);

    //Constructor
    template<size_t M>
    sparse_block_tree_node(const sequence<M,size_t>& key,size_t cur_idx);

    template<size_t M>
    void push_back(const sequence<M,size_t>& key,size_t cur_idx);

    //Must friend our iterator
    template<size_t M> 
    friend class sparse_block_tree_iterator; 
};

sparse_block_tree_node<1>::sub_key_iterator sparse_block_tree_node<1>::get_sub_key_iterator(const std::vector<size_t>& sub_key,size_t cur_idx)
{ 
    return m_keys.begin(); 
}

template<size_t M>
sparse_block_tree_node<1>::sparse_block_tree_node(const sequence<M,size_t>& key,size_t cur_idx)
{
    m_keys.push_back(key[cur_idx]);
}

template<size_t M>
void sparse_block_tree_node<1>::push_back(const sequence<M,size_t>& key,size_t cur_idx)
{
    m_keys.push_back(key[cur_idx]);
}

//TODO: RE-MAKE THINGS PRIVATE/PROTECTED AS APPRORPIATE
template<size_t N>
class sparse_block_tree_node : public sparse_block_tree_node_i {
protected:
    std::vector< sparse_block_tree_node<N-1> > m_children;
public:
    //DO NOT CALL...only for std::vector compatibility
    sparse_block_tree_node() {}; 


    //Constructor
    //Must be templated so that it can use the sequence length of the parent
    template<size_t M>
    sparse_block_tree_node(const sequence<M,size_t>& key,size_t cur_idx);

    sub_key_iterator get_sub_key_iterator(const std::vector<size_t>& sub_key,size_t cur_idx);

    template<size_t M>
    void push_back(const sequence<M,size_t>& key,size_t cur_idx);

    //Must friend our iterator
    template<size_t M> 
    friend class sparse_block_tree_iterator; 
};

//Constructor
//Must be templated so that it can use the sequence length of the parent
template<size_t N> template<size_t M>
sparse_block_tree_node<N>::sparse_block_tree_node(const sequence<M,size_t>& key,size_t cur_idx)
{
    m_keys.push_back(key[cur_idx]);
    m_children.push_back(sparse_block_tree_node<N-1>(key,cur_idx+1));
}

template<size_t N>
//typename sparse_block_tree_node<N>::sub_key_iterator sparse_block_tree_node<N>::get_sub_key_iterator(const std::vector<size_t>& sub_key,size_t cur_idx)
typename sparse_block_tree_node<N>::sub_key_iterator sparse_block_tree_node<N>::get_sub_key_iterator(const std::vector<size_t>& sub_key,size_t cur_idx)
{
    //Find the position of the current key in this node's list of keys
    size_t cur_val = sub_key[cur_idx];
    sub_key_iterator cur_pos = std::lower_bound(m_keys.begin(),m_keys.end(),cur_val);
    if(cur_pos == m_keys.end() || *cur_pos != cur_val)
    {
        throw bad_parameter(g_ns,"sparse_block_tree_node<N>","get_sub_key_iterator(...)",
            __FILE__,__LINE__,"key not found"); 
    }
    else
    {
        return m_children[distance(m_keys.begin(),cur_pos)].get_sub_key_iterator(sub_key,cur_idx+1);
    }
}

//Must be templated so that it can use the sequence length of the parent
template<size_t N> template<size_t M>
void sparse_block_tree_node<N>::push_back(const sequence<M,size_t>& key,size_t cur_idx)
{
    size_t cur_key = key[cur_idx];
    if(m_keys.back() != cur_key)
    {
        m_keys.push_back(cur_key);
        m_children.push_back(sparse_block_tree_node<N-1>(key,cur_idx+1));
    }
    else
    {
        m_children.back().push_back(key,cur_idx+1);
    }
}

//Forward declaration for specialization
template<size_t N>
class sparse_block_tree_iterator;

template<>
class sparse_block_tree_iterator<1> {
private:
    size_t m_cur_pos;
    bool m_finished;
    sparse_block_tree_node<1>& m_node;

    template<size_t M> 
    void _create_key(sequence<M,size_t>& key);
public:
    sparse_block_tree_iterator(sparse_block_tree_node<1>& node) : m_node(node),m_cur_pos(0),m_finished(false) {}

    //Prefix increment
    sparse_block_tree_iterator<1>& operator++();

    size_t& operator*();

    //Must be friends with one iterator up in order to recurse 
    template<size_t M>
    friend class sparse_block_tree_iterator;
};

sparse_block_tree_iterator<1>& sparse_block_tree_iterator<1>::operator++()
{
    m_cur_pos++;
    if(m_cur_pos ==  m_node.m_values.size())
    {
        m_finished = true;
    }
    return (*this);
}

size_t& sparse_block_tree_iterator<1>::operator*()
{
    return m_node.m_values[m_cur_pos];
}

template<size_t M>
void sparse_block_tree_iterator<1>::_create_key(sequence<M,size_t>& key)
{
    key[M-1] = m_node.m_keys[m_cur_pos];
}

template<size_t N>
class sparse_block_tree_iterator {
private:
    //TODO const qualify some of this stuff??
    size_t m_cur_pos;
    bool m_finished;
    sparse_block_tree_node<N>& m_node;

    sparse_block_tree_iterator<N-1> child;

    //Must be templated to be callable by higher-order instances 
    template<size_t M> 
    void _create_key(sequence<M,size_t>& key);
public:
    sparse_block_tree_iterator(sparse_block_tree_node<N>& node) : m_node(node),child(node.m_children[0]),m_cur_pos(0),m_finished(false) {}
    sequence<N,size_t> key();

    //Prefix increment
    sparse_block_tree_iterator<N>& operator++();

    size_t& operator*();

    //Must friend higher specialization for recursion
    template<size_t M>
    friend class sparse_block_tree_iterator;
};

template<size_t N> template<size_t M> 
void sparse_block_tree_iterator<N>::_create_key(sequence<M,size_t>& key) 
{
    key[M-N] = m_node.m_keys[m_cur_pos]; 
    child._create_key(key);
}

template<size_t N>
sequence<N,size_t> sparse_block_tree_iterator<N>::key()
{

    sequence<N,size_t> key;
    _create_key(key);
    return key;
}

template<size_t N>
sparse_block_tree_iterator<N>& sparse_block_tree_iterator<N>::operator++()
{
    if(!m_finished)
    {
        if(child.m_finished)
        {
            //Progress along this level, or finish if done
            ++m_cur_pos;
            if(m_cur_pos == m_node.m_keys.size())
            {
                m_finished = true;
            }
            else
            {
                child = sparse_block_tree_iterator<N-1>(m_node.m_children[m_cur_pos]);
            }
        }
        else
        {
            //Let the child progress along its deeper level
            ++child;
        }
    }
    return (*this);
}

template<size_t N>
size_t& sparse_block_tree_iterator<N>::operator*()
{
    return *child;
}

} // namespace impl



template<size_t N>
class sparse_block_tree : public impl::sparse_block_tree_node<N> {
public:
    typedef std::vector<size_t>::iterator sub_key_iterator;
    typedef impl::sparse_block_tree_iterator<N> iterator;
    typedef std::pair<const sequence<N,size_t>&, size_t&> value_type;

    //For iterating over the full contents
    iterator begin() { return iterator(*this); };
    iterator end();

    //Constructor
    sparse_block_tree(std::vector< sequence<N,size_t> >& sig_blocks);

    //Handles parameter validation, then passes to the base class
    sub_key_iterator get_sub_key_iterator(const std::vector<size_t>& sub_key);
};

//Arguments must be passed in lexicographic order
//Initializing with base class constructor so that every node has at least one element
//allows us to simplify base class code, have fewer error checks
template<size_t N>
sparse_block_tree<N>::sparse_block_tree(std::vector< sequence<N,size_t> >& sig_blocks) : impl::sparse_block_tree_node<N>(sig_blocks[0],0)
{
    //Ensure that block list is sorted in lexicographic order
    for(size_t i = 1; i < sig_blocks.size(); ++i)
    {
        const sequence<N,size_t>& cur = sig_blocks[i];
        const sequence<N,size_t>& prev = sig_blocks[i-1];

        bool equal = true;
        for(size_t j = 0; j < N; ++j)
        {
            if(cur[j] < prev[j])
            {
                throw bad_parameter(g_ns,"sparse_block_tree_node<N>","push_back(...)",
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
            throw bad_parameter(g_ns,"sparse_block_tree_node<N>","push_back(...)",
                __FILE__,__LINE__,"duplicate keys are not allowed"); 
        }

        push_back(cur,0);
    }
}

//Must have 0 < Key size < N
template<size_t N> 
typename sparse_block_tree<N>::sub_key_iterator sparse_block_tree<N>::get_sub_key_iterator(const std::vector<size_t>& sub_key) 
{
    if((sub_key.size() == 0) || (sub_key.size() > (N-1)))
    {
        throw bad_parameter(g_ns,"sparse_block_tree_node<N>","push_back(...)",
            __FILE__,__LINE__,"key is too long"); 
    }
    return impl::sparse_block_tree_node<N>::get_sub_key_iterator(sub_key,0); 
};

} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_H */
