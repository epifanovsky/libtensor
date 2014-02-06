#ifndef LIBTENSOR_EXPR_GRAPH_H
#define LIBTENSOR_EXPR_GRAPH_H

#include <map>
#include <vector>
#include <libtensor/exception.h>
#include <libtensor/core/noncopyable.h>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Implementation of a directed graph of nodes

    \ingroup libtensor_expr
 **/
class graph : public noncopyable {
public:
    typedef size_t node_id_t; //!< Node ID type
    typedef std::vector<node_id_t> edge_list_t; //!< Edge list type

private:
    struct vertex {
        node *data; //!< Node
        edge_list_t edges_in;  //!< Edges to node
        edge_list_t edges_out; //!< Edges from node

        vertex() : data(0) { }
    };
    typedef std::map<node_id_t, vertex> map_t;

    map_t m_lst; //!< List of vertexes

public:
    typedef map_t::const_iterator iterator;

public:
    /** \brief Constructs an empty graph
     **/
    graph() { }

    /** \brief Copy constructor
     **/
    graph(const graph &e);

    /** \brief Destructor
     **/
    virtual ~graph();

    //! \name Modifiers
    //@{

    /** \brief Adds a new node
        \return ID of newly added node
     **/
    node_id_t add(const node &n);

    /** \brief Add a new edge from id1 to id2
     **/
    void add(node_id_t id1, node_id_t id2);

    /** \brief Removes node id and all edges from or to it
     **/
    void erase(node_id_t id);

    /** \brief Remove the edge from id1 to id2, if it exists
     **/
    void erase(node_id_t id1, node_id_t id2);

    /** \brief Replaces node id with n
        \param id ID of node to replace
        \param n Node to replace i with
     **/
    void replace(node_id_t id, const node &n);

    /** \brief Replaces edge id1->id2 by id1->id3
        \param id1
        \param id2
     **/
    void replace(node_id_t id1, node_id_t id2a, node_id_t id2b);

    //@}

    //! \name Access functions
    //@{

    /** \brief Returns the number of nodes
     **/
    size_t get_n_vertexes() const {
        return m_lst.size();
    }

    /** \brief Returns STL-iterator to start of vertex list
     **/
    iterator begin() const {
        return m_lst.begin();
    }

    /** \brief Returns STL-iterator to end of vertex list
     **/
    iterator end() const {
        return m_lst.end();
    }

    /** \brief Returns node ID
     **/
    node_id_t get_id(iterator i) const {
        return i->first;
    }

    /** \brief Returns node
     **/
    const node &get_vertex(iterator i) const {
        return *(i->second.data);
    }

    /** \brief Returns list of nodes with edges to i
     **/
    const edge_list_t &get_edges_in(iterator i) const {
        return i->second.edges_in;
    }

    /** \brief Returns list of node with edges from i
     **/
    const edge_list_t &get_edges_out(iterator i) const {
        return i->second.edges_out;
    }

    /** \brief Returns the node with id
     **/
    const node &get_vertex(node_id_t id) const {
        iterator i = m_lst.find(id);
        check(i);
        return *(i->second.data);
    }

    /** \brief Returns list of nodes with edges to node id
     **/
    const edge_list_t &get_edges_in(node_id_t id) const {
        iterator i = m_lst.find(id);
        check(i);
        return i->second.edges_in;
    }

    /** \brief Returns list of node with edges from node id
     **/
    const edge_list_t &get_edges_out(node_id_t id) const {
        iterator i = m_lst.find(id);
        check(i);
        return i->second.edges_out;
    }

    //@}

    /** \brief Checks if there is a connection from id1 to id2
        \param id1 Start node
        \param id2 End node
        \return True, if connection (not necessarily direct) exists
     **/
    bool is_connected(node_id_t id1, node_id_t id2) const;

protected:
    void check(iterator i) const {
#ifdef LIBTENSOR_DEBUG
        if (i == m_lst.end()) {
            throw bad_parameter(g_ns, "graph", "check(iterator)",
                    __FILE__, __LINE__, "i.");
        }
#endif // LIBTENSOR_DEBUG
    }

    bool is_connected(iterator i1, iterator i2) const;

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_GRAPH_H
