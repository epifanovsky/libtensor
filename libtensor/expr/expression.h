#ifndef LIBTENSOR_EXPRESSION_H
#define LIBTENSOR_EXPRESSION_H

#include <vector>

namespace libtensor {


/** \brief Node of a label-free tensor expression

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class expression_node {
public:
    /** \brief Virtual destructor
     **/
    virtual ~expression_node() { }

    /** \brief Clones the node using operator new
     **/
    virtual expression_node<N, T> *clone() const = 0;

    /** \brief Returns the type of the node
     **/
    virtual const char *get_type() const = 0;

    /** \brief Applies a scaling coefficient
     **/
    virtual void scale(const T &s) = 0;

};


/** \brief Container for a label-free tensor expression

    This structure contains a list of expression nodes with an implied
    operation of addition.

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class expression {
private:
    std::vector< expression_node<N, T>* > m_nodes; //!< List of nodes

public:
    /** \brief Default constructor
     **/
    expression() { }

    /** \brief Copy constructor
     **/
    expression(const expression<N, T> &e) {
        for(size_t i = 0; i < e.m_nodes.size(); i++) add_node(*e.m_nodes[i]);
    }

    /** \brief Destructor
     **/
    ~expression() {
        for(size_t i = 0; i < m_nodes.size(); i++) delete m_nodes[i];
    }

    /** \brief Returns the list of nodes
     **/
    const std::vector< expression_node<N, T>* > &get_nodes() const {
        return m_nodes;
    }

    /** \brief Adds a node to the list
     **/
    void add_node(const expression_node<N, T> &node) {
        m_nodes.push_back(node.clone());
    }

    /** \brief Applies a scaling coefficient to every node
     **/
    void scale(const T &s) {
        for(size_t i = 0; i < m_nodes.size(); i++) m_nodes[i]->scale(s);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_EXPRESSION_H
