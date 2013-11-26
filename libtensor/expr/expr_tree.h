#ifndef LIBTENSOR_EXPR_EXPR_TREE_H
#define LIBTENSOR_EXPR_EXPR_TREE_H

#include "graph.h"

namespace libtensor {
namespace expr {


/** \brief Context-free expression

    \ingroup libtensor_expr
 **/
class expr_tree : public graph {
private:
    node_id_t m_root; //!< Root node

public:
    /** \brief Constructs an expression tree consisting of a single node
     **/
    expr_tree(const node &n) : m_root(0) {
        m_root = graph::add(n);
    }

    /** \brief Destructor
     **/
    virtual ~expr_tree() { }

    /** \brief Adds new node n as the next child of node id
        \return Id of the newly added node
     **/
    node_id_t add(node_id_t id, const node &n);

    /** \brief Adds expression tree as the next child of node id
        \return Id of the head to the newly added subtree
     **/
    node_id_t add(node_id_t id, const expr_tree &subtree);

    /** \brief Adds new node n at the position of node id
        \return Id of the newly added node

        Node id becomes the first child node of n
     **/
    node_id_t insert(node_id_t id, const node &n);

    /** \brief Deletes subtree with head node h

        h is also deleted if it is not the tree root
     **/
    void erase_subtree(node_id_t h);

    /** \brief Move subtree with head h1 to node h2 as its next child
        \return True, if successful

        The operation will only succeed if h2 is not part of h1.
     **/
    bool move(node_id_t h1, node_id_t h2);

    /** \brief Replaces subtree with head h1 by subtree with head h2
        \return True, if successful

        Subtree at h1 is deleted. The operation will only succeed if h1 is not
        part of h2.
     **/
    bool replace(node_id_t h1, node_id_t h2);

    /** \brief Copy the subtree with head node h
     **/
    expr_tree get_subtree(node_id_t h) const;

    /** \brief Returns head node of tree
     **/
    node_id_t get_root() const {
        return m_root;
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EXPR_TREE_H
