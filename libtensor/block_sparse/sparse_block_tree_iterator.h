#ifndef SPARSE_BLOCK_TREE_ITERATOR_H
#define SPARSE_BLOCK_TREE_ITERATOR_H

namespace libtensor {

namespace impl {

/* Used to allow for shared code between const and non-const iterators
 * A convenient workaround for the fact that member variables cannot be overwritten
 */
template<bool is_const>
class iterator_const_traits;

template<>
class iterator_const_traits<false> {
public:
    typedef sparse_block_tree_node* ptr_type;
    typedef sparse_block_tree_branch_node* branch_ptr_t;
    typedef sparse_block_tree_leaf_node* leaf_ptr_t;
    typedef size_t& ref_type; 
};

template<>
class iterator_const_traits<true> {
public:
    typedef const sparse_block_tree_node* ptr_type;
    typedef const sparse_block_tree_branch_node* branch_ptr_t;
    typedef const sparse_block_tree_leaf_node* leaf_ptr_t;
    typedef const size_t& ref_type; 
};


//end() is created by passing NULL ptr
template<bool is_const>
class sparse_block_tree_iterator : public iterator_const_traits<is_const> {
private:
    typedef typename iterator_const_traits<is_const>::ptr_type ptr_type;
    typedef typename iterator_const_traits<is_const>::ref_type ref_type;
    typedef typename iterator_const_traits<is_const>::branch_ptr_t branch_ptr_t;
    typedef typename iterator_const_traits<is_const>::leaf_ptr_t leaf_ptr_t;

    ptr_type m_node;
    size_t m_order;

    //TODO const qualify some of this stuff??
    size_t m_cur_pos;
    sparse_block_tree_iterator<is_const>* m_child;

    //Must be templated to be callable by higher-order instances 
    void _create_key(std::vector<size_t>& key);
public:
    //Constructor - create an iterator over all of the children of a given node, optionally starting 
    //at a given set of positions within each node. This allows iterators to be created by search() at the appropriate
    //positions in sub-linear time.
    sparse_block_tree_iterator(ptr_type node,const std::vector<size_t>& positions,const size_t order,const size_t idx = 0);

    //Copy constructor - vital for intialization in for loops
    sparse_block_tree_iterator(const sparse_block_tree_iterator& rhs);

    std::vector<size_t> key();

    //Prefix increment
    sparse_block_tree_iterator<is_const>& operator++();
    
    ref_type operator*();


    bool operator==(const sparse_block_tree_iterator<is_const>& rhs) const;
    bool operator!=(const sparse_block_tree_iterator<is_const>& rhs) const;

    //Destructor
    virtual ~sparse_block_tree_iterator() { if(m_child != NULL) { delete m_child; } }
};

//Constructor
//Passing 'NULL' is used to create the END iterator
template<bool is_const>
sparse_block_tree_iterator<is_const>::sparse_block_tree_iterator(typename sparse_block_tree_iterator<is_const>::ptr_type node,const std::vector<size_t>& positions,const size_t order,const size_t idx)
{
    m_node = node; 
    m_child = NULL; 
    m_order = order;
    m_cur_pos = positions[idx]; 

    if(m_node != NULL)
    {
        if(m_order == 2)
        {
            leaf_ptr_t leaf_ptr = static_cast<leaf_ptr_t>(static_cast<branch_ptr_t>(m_node)->m_children[positions[idx]]);
            m_child = new sparse_block_tree_iterator<is_const>(leaf_ptr,positions,m_order-1,idx+1);
        }
        else if(m_order > 2)
        {
            branch_ptr_t branch_ptr = static_cast<branch_ptr_t>(static_cast<branch_ptr_t>(m_node)->m_children[positions[idx]]);
            m_child = new sparse_block_tree_iterator<is_const>(branch_ptr,positions,m_order-1,idx+1);
        }
    }
}

//Copy constructor - vital for intialization in for loops
template<bool is_const>
sparse_block_tree_iterator<is_const>::sparse_block_tree_iterator(const sparse_block_tree_iterator<is_const>& rhs) : m_node(rhs.m_node),m_cur_pos(rhs.m_cur_pos) 
{
    if(m_node != NULL)
    {
        m_child = new sparse_block_tree_iterator<is_const>(*rhs.m_child);
    }
}

template<bool is_const>
void sparse_block_tree_iterator<is_const>::_create_key(std::vector<size_t>& key)
{
    key[key.size()-m_order] = m_node->m_keys[m_cur_pos]; 
    if(m_order > 1)
    {
        m_child->_create_key(key);
    }
}

template<bool is_const>
std::vector<size_t> sparse_block_tree_iterator<is_const>::key()
{

    std::vector<size_t> key(m_order);
    _create_key(key);
    return key;
}

template<bool is_const>
sparse_block_tree_iterator<is_const>& sparse_block_tree_iterator<is_const>::operator++()
{
    if(m_node != NULL)
    {
        //Branch?
        if(m_order > 1)
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
                    if(m_order ==  2)
                    {
                        leaf_ptr_t leaf_ptr = static_cast<leaf_ptr_t>(static_cast<branch_ptr_t>(m_node)->m_children[m_cur_pos]);
                        m_child = new sparse_block_tree_iterator<is_const>(leaf_ptr,std::vector<size_t>(m_order-1,0),m_order-1);
                    }
                    else
                    {
                        branch_ptr_t branch_ptr = static_cast<branch_ptr_t>(static_cast<branch_ptr_t>(m_node)->m_children[m_cur_pos]);
                        m_child = new sparse_block_tree_iterator<is_const>(branch_ptr,std::vector<size_t>(m_order-1,0),m_order-1);
                    }
                }
            }
        }
        else
        {
            //Leaf
            ++m_cur_pos;
            if(m_cur_pos ==  m_node->m_keys.size())
            {
                m_node = NULL;
            }

        }
    }
    return (*this);
}

template<bool is_const>
typename sparse_block_tree_iterator<is_const>::ref_type sparse_block_tree_iterator<is_const>::operator*()
{
    if(m_order == 1)
    {
        return static_cast<leaf_ptr_t>(m_node)->m_values[m_cur_pos];
    }
    else
    {
        return *(*m_child);
    }
}

template<bool is_const>
bool sparse_block_tree_iterator<is_const>::operator==(const sparse_block_tree_iterator<is_const>& rhs) const
{
    if(m_node == NULL)
    {
        return (m_node == rhs.m_node);
    }
    else
    {
        if(m_order > 1)
        {
            return (m_node == rhs.m_node) && (m_cur_pos == rhs.m_cur_pos) && (*m_child == *rhs.m_child);
        }
        else
        {
            return (m_node == rhs.m_node) && (m_cur_pos == rhs.m_cur_pos);
        }
    }
}

template<bool is_const>
bool sparse_block_tree_iterator<is_const>::operator!=(const sparse_block_tree_iterator<is_const>& rhs) const
{
    return !(*this == rhs);
}

} // namespace impl

} // namespace libtensor



#endif /* SPARSE_BLOCK_TREE_ITERATOR_H */
