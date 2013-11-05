#ifndef SPARSE_BLOCK_TREE_NODE_H
#define SPARSE_BLOCK_TREE_NODE_H

#include <vector>

namespace libtensor {

namespace impl {
typedef std::vector<size_t> block_list;

//Forward declaration for friend/specialization
template<bool is_const>
class sparse_block_tree_iterator;

//General node class
//TODO: private/protect everything etc.
class sparse_block_tree_node
{
public:
    std::vector<size_t> m_keys;
    virtual void search(const std::vector<size_t>& key,std::vector<size_t>& positions,const size_t idx) const = 0;
    virtual const block_list& get_sub_key_block_list(const std::vector<size_t>& sub_key,size_t cur_idx) const = 0;
    virtual void push_back(const std::vector<size_t>& key,size_t cur_idx) = 0;

    //Need for branch node copy constructor
    virtual sparse_block_tree_node* clone() const = 0;
    virtual ~sparse_block_tree_node() {}

    //Must friend our iterator
    template<bool is_const>
    friend class sparse_block_tree_iterator; 
};

//"Leaf" node, can store values as well as keys
class sparse_block_tree_leaf_node  : public sparse_block_tree_node 
{
protected:
    std::vector<size_t> m_values;
public:
    //Only for std::vector - do not call!!!
    sparse_block_tree_leaf_node() { };
    sparse_block_tree_leaf_node(const std::vector<size_t>& key,const size_t idx);
    const block_list& get_sub_key_block_list(const std::vector<size_t>& sub_key,size_t cur_idx) const;
    void push_back(const std::vector<size_t>& key,size_t cur_idx);
    void search(const std::vector<size_t>& key,std::vector<size_t>& positions,const size_t idx) const;

    sparse_block_tree_node* clone() const { return new sparse_block_tree_leaf_node(*this); };

    //Must friend our iterator
    template<bool is_const>
    friend class sparse_block_tree_iterator; 
};

//"Branch" node, cannot store values
class sparse_block_tree_branch_node : public sparse_block_tree_node {
protected:
    size_t m_order;
    std::vector< sparse_block_tree_node* > m_children;
public:
    void search(const std::vector<size_t>& key,std::vector<size_t>& positions,const size_t idx) const;
    //Called recursively to determine index vector corresponding to a particular key

    //DO NOT CALL...only for std::vector compatibility
    sparse_block_tree_branch_node() {}; 

    //Constructor
    sparse_block_tree_branch_node(const std::vector<size_t>& key,size_t cur_idx,size_t order);

    //Copy constructor
    sparse_block_tree_branch_node(const sparse_block_tree_branch_node& rhs);

    //Assignment operator
    sparse_block_tree_branch_node& operator=(const sparse_block_tree_branch_node& rhs);

    const block_list& get_sub_key_block_list(const std::vector<size_t>& sub_key,size_t cur_idx) const;

    void push_back(const std::vector<size_t>& key,size_t cur_idx);

    sparse_block_tree_node* clone() const { return new sparse_block_tree_branch_node(*this); };


    virtual ~sparse_block_tree_branch_node();

    //Must friend our iterator
    template<bool is_const>
    friend class sparse_block_tree_iterator; 
};

} // namespace impl

} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_NODE_H */
