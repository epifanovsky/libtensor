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

    std::set<size_t> s1, s2;
    s1.insert(i);

    while (! s1.empty()) {

        bool found = false;
        std::set<size_t>::iterator ic = s1.begin();

        node_list_t::const_iterator end = m_lst.find(*ic);
        for (node_list_t::const_iterator it = m_lst.begin(); it != end; it++) {

            if (it->second.count(*ic) == 0) continue;

            found = true;
            if (s2.count(it->first) == 0) s1.insert(it->first);
        }

        if (end != m_lst.end()) {
            const adjacent_list_t &lst = end->second;
            for (adjacent_list_t::const_iterator it = lst.begin();
                    it != lst.end(); it++) {

                if (s2.count(it->first) == 0) s1.insert(it->first);
            }
            found = true;
        }

        if (found) {
            s2.insert(*ic);
        }
        s1.erase(ic);
    }

    for (std::set<size_t>::iterator it = s2.begin(); it != s2.end(); it++) {
        clst.push_back(*it);
    }
}




} // namespace libtensor

