#include <set>
#include "adjacency_list.h"


namespace libtensor {


void adjacency_list::add(size_t i, size_t j, size_t weight) {

    if (weight == 0) return;
    if (i > j) std::swap(i, j);

    m_lst[i][j] += weight;
}


void adjacency_list::erase(size_t i, size_t j) {

    if (i > j) std::swap(i, j);

    node_list_t::iterator ii = m_lst.find(i);
    if (ii == m_lst.end()) return;

    adjacent_list_t::iterator ij = ii->second.find(j);
    if (ij == ii->second.end()) return;

    ii->second.erase(ij);
    if (ii->second.empty()) m_lst.erase(ii);
}


bool adjacency_list::exist(size_t i, size_t j) const {

    return weight(i, j) != 0;
}


size_t adjacency_list::weight(size_t i, size_t j) const {

    if (i > j) std::swap(i, j);

    node_list_t::const_iterator ii = m_lst.find(i);
    if (ii == m_lst.end()) return 0;

    adjacent_list_t::const_iterator ij = ii->second.find(j);
    if (ij == ii->second.end()) return 0;

    return ij->second;
}


void adjacency_list::get_prev_neighbours(
        size_t i, std::vector<size_t> &nlst) const {

    nlst.clear();
    node_list_t::const_iterator ii = m_lst.begin();
    for (; ii != m_lst.end() && ii->first < i; ii++) {

        if (ii->second.count(i) != 0) nlst.push_back(ii->first);
    }
}


void adjacency_list::get_next_neighbours(
        size_t i, std::vector<size_t> &nlst) const {

    nlst.clear();
    node_list_t::const_iterator ii = m_lst.find(i);
    if (ii == m_lst.end()) return;

    const adjacent_list_t &lst = ii->second;
    for (adjacent_list_t::const_iterator ij = lst.begin();
            ij != lst.end(); ij++) {
        nlst.push_back(ij->first);
    }
}


void adjacency_list::get_neighbours(size_t i, std::vector<size_t> &nlst) const {

    nlst.clear();
    get_prev_neighbours(i, nlst);

    std::vector<size_t> tmplst;
    get_next_neighbours(i, tmplst);
    nlst.insert(nlst.end(), tmplst.begin(), tmplst.end());
}


void adjacency_list::get_connected(size_t i, std::vector<size_t> &clst) const {

    clst.clear();
    // Find smallest node index connected to i
    size_t imin = i;
    while (true) {
        node_list_t::const_iterator ii = m_lst.begin();
        for (; ii != m_lst.end() && ii->first < imin; ii++) {

            if (ii->second.count(imin) != 0) break;
        }

        if (ii == m_lst.end() || ii->first >= imin) break;
        imin = ii->first;
    }

    // Node i does not exist in graph
    if (m_lst.count(imin) == 0) return;

    std::set<size_t> tmp;
    tmp.insert(imin);

    std::set<size_t>::iterator it = tmp.begin();
    while (it != tmp.end()) {

        node_list_t::const_iterator ii = m_lst.find(*it);
        if (ii != m_lst.end()) {
            const adjacent_list_t &alst = ii->second;
            for (adjacent_list_t::const_iterator ij = alst.begin();
                    ij != alst.end(); ij++) {

                tmp.insert(ij->first);
            }

            it = tmp.find(ii->first);
        }
        it++;
    }

    for (std::set<size_t>::iterator it = tmp.begin(); it != tmp.end(); it++) {
        clst.push_back(*it);
    }
}




} // namespace libtensor

