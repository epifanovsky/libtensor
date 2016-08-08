#include <cstring>
#include <algorithm>
#ifdef HAVE_EXECINFO_BACKTRACE
#include <execinfo.h>
#endif // HAVE_EXECINFO_BACKTRACE
#include "backtrace.h"

namespace libutil {


backtrace::backtrace() : m_nframes(0) {

#ifdef HAVE_EXECINFO_BACKTRACE
    void *frames[256];
    char **symbols = 0;
    size_t nframes = ::backtrace(frames, 256);
    symbols = ::backtrace_symbols(frames, nframes);
    if(symbols == 0) return;

    size_t off = 0;
    nframes = std::min(nframes, size_t(k_maxframes));
    for(size_t i = 0; i < nframes; i++) {
        size_t len = strlen(symbols[i]);
        if(off + len + 1 > k_buflen) break;
        strcpy(m_buf + off, symbols[i]);
        m_frames[m_nframes++] = m_buf + off;
        off += len + 1;
    }
    free(symbols);
#endif // HAVE_EXECINFO_BACKTRACE
}


backtrace::backtrace(const backtrace &bt) {

    ::memcpy(m_buf, bt.m_buf, k_buflen);
    m_nframes = bt.m_nframes;
    for(size_t i = 0; i < m_nframes; i++) {
        m_frames[i] = m_buf + (bt.m_frames[i] - bt.m_buf);
    }
}


} // namespace libutil
