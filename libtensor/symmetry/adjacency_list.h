#ifndef LIBTENSOR_ADJACENCY_LIST_H
#define LIBTENSOR_ADJACENCY_LIST_H

#include <map>
#include <vector>

namespace libtensor {


/** \brief Adjacency list for an undirected, weighted graph
 **/
class adjacency_list {
private:
    typedef std::map<size_t, size_t> adjacent_list_t;
    typedef std::map<size_t, adjacent_list_t> node_list_t;

    node_list_t m_lst;

public:
    /** \brief Add an edge from node i to node j with weight.
        \param i First node
        \param j Second node
        \param weight Weight of edge

        If the edge already exists, the weight is added to it
     **/
    void add(size_t i, size_t j, size_t weight = 1);


    /** \brief Deletes an edge if it exists
        \param i First node
        \param j Second node
     **/
    void erase(size_t i, size_t j);


    /** \brief Empty adjacency list
     **/
    void clear() { m_lst.clear(); }


    /** \brief Returns true if edge exists
        \param i First node
        \param j Second node
     **/
    bool exist(size_t i, size_t j) const;


    /** \brief Returns the weight of an edge.
        \param i First node
        \param j Second node

        Returns zero if the weight does not exist.
     **/
    size_t weight(size_t i, size_t j) const;


    /** \brief Return the direct neighbours j of node i which have j < i
        \param i Node
        \param[out] nlst Neighbour list

        Returns empty list if node i does not exists or if it does not have
        neighbours with j < i.
     **/
    void get_prev_neighbours(size_t i, std::vector<size_t> &nlst) const;


    /** \brief Return the direct neighbours j of node i which have i < j
        \param i Node
        \param[out] nlst Neighbour list

        Returns empty list if node i does not exists or if it does not have
        neighbours with i < j
     **/
    void get_next_neighbours(size_t i, std::vector<size_t> &nlst) const;


    /** \brief Return all direct neighbours of node i
        \param i Node
        \param[out] nlst Neighbour list

        Returns empty list if node i does not exists or if it does not have
        neighbours
     **/
    void get_neighbours(size_t i, std::vector<size_t> &nlst) const;


    /** \brief Return list of xall nodes connected to i (including i)
        \param i Node
        \param[out] clst List of connected nodes

        Returns empty list if node i does not exists.
     **/
    void get_connected(size_t i, std::vector<size_t> &clst) const;
};


} // namespace libtensor


#endif // LIBTENSOR_ADJACENCY_LIST_H
