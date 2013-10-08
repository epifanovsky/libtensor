#ifndef SPARSE_LOOP_LIST_NODE_H
#define SPARSE_LOOP_LIST_NODE_H

class sparse_loop_list_node : public loop_list_node_i 
{
    virtual size_t weight() const = 0;
    virtual size_t stepa(size_t i) = 0;
    virtual size_t stepb(size_t i) = 0;

}

#endif /* SPARSE_LOOP_LIST_NODE_H */
