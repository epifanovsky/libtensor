#include <set>
#include "graph.h"

namespace libtensor {
namespace expr {


graph::graph(const graph &e) {

    for(map_t::const_iterator i = e.m_lst.begin(); i != e.m_lst.end(); ++i) {
        std::pair<map_t::iterator, bool> inew =
            m_lst.insert(map_t::value_type(i->first, vertex()));

        vertex &v = inew.first->second;
        if(!inew.second) delete v.data;
        v.data = i->second.data->clone();
        v.edges_in  = i->second.edges_in;
        v.edges_out = i->second.edges_out;
    }
}


graph::~graph() {

    for(map_t::iterator i = m_lst.begin(); i != m_lst.end(); ++i) {
        delete i->second.data;
        i->second.data = 0;
    }
}


graph::node_id_t graph::add(const node &n) {

    node_id_t id = (m_lst.empty() ? 0 : m_lst.rbegin()->first + 1);
    m_lst[id].data = n.clone();
    return id;
}


void graph::add(node_id_t id1, node_id_t id2) {

    map_t::iterator i1 = m_lst.find(id1), i2 = m_lst.find(id2);
    check(i1);
    check(i2);

    i1->second.edges_out.push_back(id2);
    i2->second.edges_in.push_back(id1);
}


void graph::erase(node_id_t id) {

    map_t::iterator i = m_lst.find(id);
    check(i);

    vertex &v = i->second;
    delete v.data; v.data = 0;
    for(edge_list_t::iterator j = v.edges_in.begin(); j != v.edges_in.end();
        ++j) {

        edge_list_t &e = m_lst.find(*j)->second.edges_out;
        for(edge_list_t::iterator k = e.begin(); k != e.end(); ++k) {
            if(*k == id) { e.erase(k); break; }
        }
    }
    for(edge_list_t::iterator j = v.edges_out.begin(); j != v.edges_out.end();
        ++j) {

        edge_list_t &e = m_lst.find(*j)->second.edges_in;
        for(edge_list_t::iterator k = e.begin(); k != e.end(); ++k) {
            if(*k == id) { e.erase(k); break; }
        }
    }
    m_lst.erase(i);
}


void graph::erase(node_id_t id1, node_id_t id2) {

    map_t::iterator i1 = m_lst.find(id1), i2 = m_lst.find(id2);
    check(i1);
    check(i2);

    edge_list_t &out = i1->second.edges_out;
    for(edge_list_t::iterator j = out.begin(); j != out.end(); ++j) {
        if(*j == id2) { out.erase(j); break; }
    }

    edge_list_t &in  = i2->second.edges_in;
    for(edge_list_t::iterator j = in.begin(); j != in.end(); ++j) {
        if(*j == id1) { in.erase(j); break; }
    }
}


void graph::replace(node_id_t id, const node &n) {

    map_t::iterator i = m_lst.find(id);
    check(i);

    delete i->second.data;
    i->second.data = n.clone();
}


void graph::replace(node_id_t id1, node_id_t id2, node_id_t id3) {

    map_t::iterator i1 = m_lst.find(id1);
    map_t::iterator i2 = m_lst.find(id2);
    map_t::iterator i3 = m_lst.find(id3);
    check(i1);
    check(i2);
    check(i3);

    edge_list_t &out = i1->second.edges_out;
    for(edge_list_t::iterator j = out.begin(); j != out.end(); ++j) {
        if(*j == id2) { *j = id3; break; }
    }

    edge_list_t &in  = i2->second.edges_in;
    for(edge_list_t::iterator j = in.begin(); j != in.end(); ++j) {
        if(*j == id1) { in.erase(j); break; }
    }

    i3->second.edges_in.push_back(id1);
}


bool graph::is_connected(node_id_t id1, node_id_t id2) const {

    iterator i1 = m_lst.find(id1), i2 = m_lst.find(id2);
    check(i1);
    check(i2);

    return is_connected(i1, i2);
}

bool graph::is_connected(iterator i1, iterator i2) const {

    if(i1 == i2) return true;

    const edge_list_t &e = i2->second.edges_in;
    for(size_t i = 0; i < e.size(); i++) {
        iterator ic = m_lst.find(e[i]);
        if(is_connected(i1, ic)) return true;
    }
    return false;
}


} // namespace expr
} // namespace libtensor
