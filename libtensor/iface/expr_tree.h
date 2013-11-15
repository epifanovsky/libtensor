#ifndef LIBTENSOR_IFACE_EXPR_TREE_H
#define LIBTENSOR_IFACE_EXPR_TREE_H

#include <libtensor/expr/graph.h>

namespace libtensor {
namespace iface {


/** \brief Context-free expression

    \ingroup libtensor_iface
 **/
class expr_tree : public expr::graph {
private:
    node_id_t m_root; //!< Root node

public:
    /** \brief Constructs an expression tree consisting of a single node
     **/
    expr_tree(const expr::node &n) : m_root(0) {
        m_root = expr::graph::add(n);
    }

    /** \brief Destructor
     **/
    virtual ~expr_tree() { }

    /** \brief Adds new node n as the next child of node id
     **/
    void add(node_id_t id, const expr::node &n);

    /** \brief Adds expression tree as the next child of node id
     **/
    void add(node_id_t id, const expr_tree &subtree);

    /** \brief Adds new node n at the position of node id

        Node id becomes the first child node of n
     **/
    void insert(node_id_t id, const expr::node &n);

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

    /** \brief Returns head node of tree
     **/
    node_id_t get_root() const {
        return m_root;
    }
};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_EXPR_TREE_H
