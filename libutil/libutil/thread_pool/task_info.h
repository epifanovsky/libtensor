#ifndef LIBUTIL_TASK_INFO_H
#define LIBUTIL_TASK_INFO_H

namespace libutil {


class task_source;
class task_i;


struct task_info {

    task_source *tsrc;
    task_i *tsk;

};


} // namespace libutil

#endif // LIBUTIL_TASK_INFO_H

