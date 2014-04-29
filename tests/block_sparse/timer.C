#include <sys/time.h>

double read_timer()
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, 0 );
        initialized = true;
    }

    gettimeofday( &end, 0 );

    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}
