#ifndef SPARSE_BLOCK_TREE_H
#define SPARSE_BLOCK_TREE_H

#include <vector>
#include <algorithm>
#include <utility>
#include "../core/sequence.h"
#include "../core/permutation.h"

//TODO REMOVE
#include <iostream>

namespace libtensor {

namespace impl {

//Forward declaration for friend/specialization
template<size_t N,bool is_const>
class sparse_block_tree_iterator;

//Forward declaration to allow specialization
template<size_t N>
class sparse_block_tree_node;

//"Leaf" node, can store values as well as keys
template<>
class sparse_block_tree_node<1> {
protected:
    std::vector<size_t> m_keys;
    std::vector<size_t> m_values;

    //Called recursively to determine index vector corresponding to a particular key
    //Must instead of sequence for key because some key sizes are determined at runtime
    void search(const std::vector<size_t>& key,std::vector<size_t>& positions,const size_t idx) const;
public:
    typedef std::vector<size_t>::iterator sub_key_iterator;
    //DO NOT CALL...only for std::vector compatibility
    sparse_block_tree_node() {}; 

    //Base case for the recursion
    sub_key_iterator get_sub_key_iterator(const std::vector<size_t>& sub_key,size_t cur_idx);

    //Constructor
    template<size_t M>
    sparse_block_tree_node(const sequence<M,size_t>& key,size_t cur_idx);

    template<size_t M>
    void push_back(const sequence<M,size_t>& key,size_t cur_idx);

    //Friend higher level nodes for recursion 
    template<size_t M> 
    friend class sparse_block_tree_node; 

    //Must friend our iterator
    template<size_t M,bool is_const>
    friend class sparse_block_tree_iterator; 
};

inline void sparse_block_tree_node<1>::search(const std::vector<size_t>& key,std::vector<size_t>& positions,const size_t idx) const
{
    std::vector<size_t>::const_iterator cur_pos_it = std::lower_bound(m_keys.begin(),m_keys.end(),key[idx]);
    if(cur_pos_it == m_keys.end() || *cur_pos_it != key[idx])
    {
        throw bad_parameter(g_ns,"sparse_block_tree_node<N>","search(...)",
            __FILE__,__LINE__,"key not found"); 
    }
    size_t cur_pos = distance(m_keys.begin(),cur_pos_it);
    positions[idx] = cur_pos;
}

inline sparse_block_tree_node<1>::sub_key_iterator sparse_block_tree_node<1>::get_sub_key_iterator(const std::vector<size_t>& sub_key,size_t cur_idx)
{ 
    return m_keys.begin(); 
}

template<size_t M>
inline sparse_block_tree_node<1>::sparse_block_tree_node(const sequence<M,size_t>& key,size_t cur_idx)
{
    m_keys.push_back(key[cur_idx]);
    m_values.push_back(0);
}

template<size_t M>
inline void sparse_block_tree_node<1>::push_back(const sequence<M,size_t>& key,size_t cur_idx)
{
    m_keys.push_back(key[cur_idx]);
    m_values.push_back(0);
}

//TODO: RE-MAKE THINGS PRIVATE/PROTECTED AS APPRORPIATE
template<size_t N>
class sparse_block_tree_node {
protected:
    std::vector<size_t> m_keys;
    std::vector< sparse_block_tree_node<N-1> > m_children;

    //Called recursively to determine index vector corresponding to a particular key
    void search(const std::vector<size_t>& key,std::vector<size_t>& positionsm,const size_t idx) const;
public:
    typedef std::vector<size_t>::iterator sub_key_iterator;

    //DO NOT CALL...only for std::vector compatibility
    sparse_block_tree_node() {}; 


    //Constructor
    //Must be templated so that it can use the sequence length of the parent
    template<size_t M>
    sparse_block_tree_node(const sequence<M,size_t>& key,size_t cur_idx);

    sub_key_iterator get_sub_key_iterator(const std::vector<size_t>& sub_key,size_t cur_idx);

    template<size_t M>
    void push_back(const sequence<M,size_t>& key,size_t cur_idx);

    //Friend higher level nodes for recursion 
    template<size_t M> 
    friend class sparse_block_tree_node; 

    //Must friend our iterator
    template<size_t M,bool is_const>
    friend class sparse_block_tree_iterator; 
};

template<size_t N>
void sparse_block_tree_node<N>::search(const std::vector<size_t>& key,std::vector<size_t>& positions,const size_t idx) const
{
    std::vector<size_t>::const_iterator cur_pos_it = std::lower_bound(m_keys.begin(),m_keys.end(),key[idx]);
    if(cur_pos_it == m_keys.end() || *cur_pos_it != key[idx])
    {
        throw bad_parameter(g_ns,"sparse_block_tree_node<N>","search(...)",
            __FILE__,__LINE__,"key not found"); 
    }
    size_t cur_pos = distance(m_keys.begin(),cur_pos_it);
    positions[idx] = cur_pos;
    m_children[cur_pos].search(key,positions,idx+1);
}


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

/* Used to allow for shared code between const and non-const iterators
 * A convenient workaround for the fact that member variables cannot be overwritten
 */
template<size_t N,bool is_const>
class iterator_const_traits;

template<size_t N>
class iterator_const_traits<N,false> {
protected:
    typedef sparse_block_tree_node<N>* ptr_type;
    typedef size_t& ref_type; 
};

template<size_t N>
class iterator_const_traits<N,true> {
protected:
    typedef const sparse_block_tree_node<N>* ptr_type;
    typedef const size_t& ref_type; 
};

template<bool is_const>
class sparse_block_tree_iterator<1,is_const> : public iterator_const_traits<1,is_const> {
private:
    size_t m_cur_pos;
    typedef typename iterator_const_traits<1,is_const>::ptr_type ptr_type;
    typedef typename iterator_const_traits<1,is_const>::ref_type ref_type;
    ptr_type m_node;
    

    template<size_t M> 
    void _create_key(sequence<M,size_t>& key);
public:
    //TODO: Make this comment more clear
    //Constructor - create an iterator over all of the children of a given node,starting 
    //at a given set of positions within each node. This allows iterators to be created by search() at the appropriate
    //positions in sub-linear time. 
    sparse_block_tree_iterator(ptr_type node,const std::vector<size_t>& positions,const size_t idx = 0);

    //Copy constructor - vital for intialization in for loops
    sparse_block_tree_iterator(const sparse_block_tree_iterator<1,is_const>& rhs) : m_node(rhs.m_node),m_cur_pos(rhs.m_cur_pos) {}


    //Prefix increment
    sparse_block_tree_iterator<1,is_const>& operator++();

    ref_type operator*();

    bool operator==(const sparse_block_tree_iterator<1,is_const>& rhs) const;
    bool operator!=(const sparse_block_tree_iterator<1,is_const>& rhs) const;
    
    //Must be friends with one iterator up in order to recurse 
    template<size_t M,bool other_is_const>
    friend class sparse_block_tree_iterator;
};

template<bool is_const>
inline sparse_block_tree_iterator<1,is_const>::sparse_block_tree_iterator(sparse_block_tree_iterator<1,is_const>::ptr_type node,const std::vector<size_t>& positions,const size_t idx)
{
    m_cur_pos = positions[idx];
    m_node = node;
}

template<bool is_const> template<size_t M>
inline void sparse_block_tree_iterator<1,is_const>::_create_key(sequence<M,size_t>& key)
{
    key[M-1] = m_node->m_keys[m_cur_pos];
}

template<bool is_const>
inline sparse_block_tree_iterator<1,is_const>& sparse_block_tree_iterator<1,is_const>::operator++()
{
    if(m_node != NULL)
    {
        m_cur_pos++;
        if(m_cur_pos ==  m_node->m_keys.size())
        {
            m_node = NULL;
        }
    }
    return (*this);
}

template<bool is_const>
inline typename sparse_block_tree_iterator<1,is_const>::ref_type sparse_block_tree_iterator<1,is_const>::operator*()
{
    return m_node->m_values[m_cur_pos];
}

template<bool is_const>
inline bool sparse_block_tree_iterator<1,is_const>::operator==(const sparse_block_tree_iterator<1,is_const>& rhs) const
{
    return (m_node == rhs.m_node) && (m_cur_pos == rhs.m_cur_pos);
}

template<bool is_const>
inline bool sparse_block_tree_iterator<1,is_const>::operator!=(const sparse_block_tree_iterator<1,is_const>& rhs) const
{
    return !(*this != rhs);
}

//TODO: Interoperability of iterator and const_iterator
//TODO: formally add std::iterator/iterator traits biznas...
//Type is templated so as not to duplicate code between iterator and const iterator
template<size_t N,bool is_const>
class sparse_block_tree_iterator : public iterator_const_traits<N,is_const> {
private:
    typedef typename iterator_const_traits<N,is_const>::ptr_type ptr_type;
    typedef typename iterator_const_traits<N,is_const>::ref_type ref_type;

    ptr_type m_node;

    //TODO const qualify some of this stuff??
    size_t m_cur_pos;
    sparse_block_tree_iterator<N-1,is_const>* m_child;

    //Must be templated to be callable by higher-order instances 
    template<size_t M> 
    void _create_key(sequence<M,size_t>& key);
public:
    //Constructor - create an iterator over all of the children of a given node, optionally starting 
    //at a given set of positions within each node. This allows iterators to be created by search() at the appropriate
    //positions in sub-linear time.
    sparse_block_tree_iterator(ptr_type node,const std::vector<size_t>& positions,const size_t idx = 0);

    //Copy constructor - vital for intialization in for loops
    sparse_block_tree_iterator(const sparse_block_tree_iterator& rhs);

    sequence<N,size_t> key();

    //Prefix increment
    sparse_block_tree_iterator<N,is_const>& operator++();
    
    ref_type operator*();

    //Must friend higher specialization for recursion
    template<size_t M,bool other_is_const>
    friend class sparse_block_tree_iterator;

    bool operator==(const sparse_block_tree_iterator<N,is_const>& rhs) const;
    bool operator!=(const sparse_block_tree_iterator<N,is_const>& rhs) const;

    //Destructor
    virtual ~sparse_block_tree_iterator() { if(m_node != NULL) { delete m_child; } }
};

//Constructor
//Passing 'NULL' is used to create the END iterator
template<size_t N,bool is_const>
sparse_block_tree_iterator<N,is_const>::sparse_block_tree_iterator(ptr_type node,const std::vector<size_t>& positions,const size_t idx) : m_node(node)
{
    m_cur_pos =  positions[idx];
    if(m_node != NULL)
    {
        m_child = new sparse_block_tree_iterator<N-1,is_const>(&node->m_children[positions[idx]],positions,idx+1);
    }
}

//Copy constructor - vital for intialization in for loops
template<size_t N,bool is_const>
sparse_block_tree_iterator<N,is_const>::sparse_block_tree_iterator(const sparse_block_tree_iterator<N,is_const>& rhs) : m_node(rhs.m_node),m_cur_pos(rhs.m_cur_pos) 
{
    if(m_node != NULL)
    {
        m_child = new sparse_block_tree_iterator<N-1,is_const>(*rhs.m_child);
    }
}

template<size_t N,bool is_const> template<size_t M>
void sparse_block_tree_iterator<N,is_const>::_create_key(sequence<M,size_t>& key) 
{
    key[M-N] = m_node->m_keys[m_cur_pos]; 
    m_child->_create_key(key);
}

template<size_t N,bool is_const>
sequence<N,size_t> sparse_block_tree_iterator<N,is_const>::key()
{

    sequence<N,size_t> key;
    _create_key(key);
    return key;
}

template<size_t N,bool is_const>
sparse_block_tree_iterator<N,is_const>& sparse_block_tree_iterator<N,is_const>::operator++()
{
    if(m_node != NULL)
    {
        //Try to progress the inner node
        ++(*m_child);
        //If that moved us to the end of the child node, advance this  node
        if(m_child->m_node == NULL)
        {
            //Progress along this level, or finish if done
            ++m_cur_pos;
            if(m_cur_pos == m_node->m_keys.size())
            {
                m_node = NULL;
            }
            else
            {
                delete m_child;
                m_child = new sparse_block_tree_iterator<N-1,is_const>(&m_node->m_children[m_cur_pos],std::vector<size_t>(N-1,0));
            }
        }
    }
    return (*this);
}

template<size_t N,bool is_const>
typename sparse_block_tree_iterator<N,is_const>::ref_type sparse_block_tree_iterator<N,is_const>::operator*()
{
    return *(*m_child);
}

template<size_t N,bool is_const>
bool sparse_block_tree_iterator<N,is_const>::operator==(const sparse_block_tree_iterator<N,is_const>& rhs) const
{
    if(m_node == NULL)
    {
        return (m_node == rhs.m_node);
    }
    else
    {
        return (m_node == rhs.m_node) && (m_cur_pos == rhs.m_cur_pos) && (m_child == rhs.m_child);
    }
}

template<size_t N,bool is_const>
bool sparse_block_tree_iterator<N,is_const>::operator!=(const sparse_block_tree_iterator<N,is_const>& rhs) const
{
    return !(*this == rhs);
}


} // namespace impl



template<size_t N>
class sparse_block_tree : public impl::sparse_block_tree_node<N> {
public:
    typedef std::vector<size_t>::iterator sub_key_iterator;
    typedef impl::sparse_block_tree_iterator<N,false> iterator;
    typedef impl::sparse_block_tree_iterator<N,true> const_iterator;

    //Constructor
    sparse_block_tree(const std::vector< sequence<N,size_t> >& sig_blocks);

    //For iterating over the full contents
    iterator begin() { return iterator(this,std::vector<size_t>(N,0)); };
    const_iterator begin() const { return const_iterator(this,std::vector<size_t>(N,0)); }
    iterator end() { return iterator(NULL,std::vector<size_t>(N,0)); };
    const_iterator end() const { return const_iterator(NULL,std::vector<size_t>(N,0)); }

    //Searching for a specific key
    //Use a vector because sometimes key lengths must be determined at runtime
    iterator search(const std::vector<size_t>& key);
    const_iterator search(const std::vector<size_t>& key) const;

    //Get the iterator over the sub keys of a given key
    sub_key_iterator get_sub_key_iterator(const std::vector<size_t>& sub_key);

    //Can't use permutation<N> class because permutation degree may need to be determined at runtime
    sparse_block_tree<N> permute(permutation<N>& perm);

    bool operator==(const sparse_block_tree<N>& rhs) const;
    bool operator!=(const sparse_block_tree<N>& rhs) const;
};

//Arguments must be passed in lexicographic order
//Initializing with base class constructor so that every node has at least one element
//allows us to simplify base class code, have fewer error checks
template<size_t N>
sparse_block_tree<N>::sparse_block_tree(const std::vector< sequence<N,size_t> >& sig_blocks) : impl::sparse_block_tree_node<N>(sig_blocks[0],0)
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
                throw bad_parameter(g_ns,"sparse_block_tree<N>","sparse_block_tree(...)",
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

        push_back(cur,0);
    }
}

template<size_t N>
typename sparse_block_tree<N>::iterator sparse_block_tree<N>::search(const std::vector<size_t>& key)
{
    if(key.size() != N)
    {
        throw bad_parameter(g_ns,"sparse_block_tree<N>","search(...)",
            __FILE__,__LINE__,"key length does not match depth of tree"); 
    }
    std::vector<size_t> positions(N);
    impl::sparse_block_tree_node<N>::search(key,positions,0);
    return iterator(this,positions);
}

template<size_t N>
typename sparse_block_tree<N>::const_iterator sparse_block_tree<N>::search(const std::vector<size_t>& key) const
{
    if(key.size() != N)
    {
        throw bad_parameter(g_ns,"sparse_block_tree<N>","search(...)",
            __FILE__,__LINE__,"key length does not match depth of tree");
    }

    std::vector<size_t> positions(N);
    impl::sparse_block_tree_node<N>::search(key,positions,0);
    return const_iterator(this,positions);
}

namespace impl { 

template<size_t M>
struct seq_val_compare {
    bool operator()(const std::pair< sequence<M,size_t>,size_t>& p1,const std::pair< sequence<M,size_t>,size_t>& p2)
    {

        for(size_t i = 0; i < M; ++i)
        {
            if(p1.first[i] < p2.first[i])
            {
                return true;
            }
            else if(p1.first[i] > p2.first[i])
            {
                return false;
            }
        }
        return false;
    }
};

} // namespace impl

template<size_t N>
sparse_block_tree<N> sparse_block_tree<N>::permute()
{
    std::vector< std::pair< sequence<N,size_t>, size_t > > kv_pairs;
    for(iterator it = begin(); it != end(); ++it)
    {
        sequence<N,size_t> new_key = it.key(); 
        perm.apply(new_key);
        kv_pairs.push_back(std::make_pair(new_key,*it));
    }


    std::sort(kv_pairs.begin(),kv_pairs.end(),impl::seq_val_compare<N>());

    std::vector< sequence<N,size_t> > all_keys;
    std::vector<size_t> all_vals;
    for(size_t i = 0; i < kv_pairs.size(); ++i)
    {
        all_keys.push_back(kv_pairs[i].first);
        all_vals.push_back(kv_pairs[i].second);
    }

    sparse_block_tree<N> sbt(all_keys);
    size_t m = 0; 
    for(iterator it = sbt.begin(); it != sbt.end(); ++it)
    {
        *it = all_vals[m];
        ++m;
    }
    return sbt;
}
template<size_t N>
bool sparse_block_tree<N>::operator==(const sparse_block_tree<N>& rhs) const
{
    typename sparse_block_tree<N>::const_iterator rhs_it = rhs.begin();
    for(typename sparse_block_tree<N>::const_iterator lhs_it = begin(); lhs_it != end(); ++lhs_it)
    {
        if(rhs_it == rhs.end())
        {
            return false;
        }

        //Compare keys
        const sequence<N,size_t>& lhs_key = lhs_it.key();
        const sequence<N,size_t>& rhs_key = rhs_it.key();

        for(size_t i = 0; i < N; ++i)
        {
            if(lhs_key[i] != rhs_key[i])
            {
                return false;
            }
        }

        //Compare values
        if(*lhs_it != *rhs_it)
        {
            return false;
        }
        ++rhs_it;
    }
    return true;
}

template<size_t N>
bool sparse_block_tree<N>::operator!=(const sparse_block_tree<N>& rhs) const
{
    return !(*this == rhs);
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

//Type erasure class used to hide order for when a container must contain sparse_block_trees of varying orders 
class sparse_block_tree_any_order {
private:
    class abstract_base {
    public:
        virtual abstract_base* clone() const = 0;
        virtual ~abstract_base() {}
        
        //Returns the value at the specified key 
        virtual size_t search(const std::vector<size_t>& key) const = 0;
        virtual bool equal(const abstract_base* rhs) const = 0; 
    };

    template<size_t N>
    class wrapper : public abstract_base {
    private:
        sparse_block_tree<N> m_tree;
    public:
        size_t search(const std::vector<size_t>& key) const { return *(m_tree.search(key)); }
        bool equal(const abstract_base* rhs) const { return m_tree == static_cast<const wrapper<N>* >(rhs)->m_tree; }
        wrapper(const sparse_block_tree<N>& tree) : m_tree(tree) {}
        abstract_base* clone() const { return new wrapper<N>(m_tree); }
    };

    //Instance variables
    abstract_base* m_tree_ptr;
    size_t m_order;
public:

    //Constructor
    template<size_t N>
    sparse_block_tree_any_order(const sparse_block_tree<N>& sbt) : m_order(N), m_tree_ptr(new wrapper<N>(sbt)) {}

    //Copy constructor
    sparse_block_tree_any_order(const sparse_block_tree_any_order& rhs) : m_order(rhs.m_order), m_tree_ptr(rhs.m_tree_ptr->clone()) {}
    
    //Assignment operator
    sparse_block_tree_any_order& operator=(const sparse_block_tree_any_order& rhs) { m_order = rhs.m_order; m_tree_ptr = rhs.m_tree_ptr->clone(); return *this; }
    
    //Destructor
    virtual ~sparse_block_tree_any_order() { delete m_tree_ptr; }

    //Returns the value at the specified key 
    size_t search(const std::vector<size_t>& key) const { return m_tree_ptr->search(key); };

    bool operator==(const sparse_block_tree_any_order& rhs) const { if(m_order != rhs.m_order) { return false; } return m_tree_ptr->equal(rhs.m_tree_ptr); }
    bool operator!=(const sparse_block_tree_any_order& rhs) const { return !(*this == rhs); }

    //Accessors
    size_t get_order() const { return m_order; }
    abstract_base* get_ptr() { return m_tree_ptr; }
};

} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_H */
