#if 0
//XXX mpi
#ifdef WITH_MPI
#include <mpi.h>
#endif

#ifdef WITH_MPI
#define MPI_CHUNK_SIZE 128
#endif

#ifdef WITH_MPI
static void
master(int size)
{
	MPI_Status status;
	int range[2];

	range[1] = 0;

	while (have_tasks()) {
		MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

		range[0] = range[1];

		for (int i = 0; i < MPI_CHUNK_SIZE && have_tasks(); i++) {
			range[1]++;
			task_next();
		}

		MPI_Send(range, 2, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
	}

	range[0] = range[1] = -1;

	for (int i = 1; i < size; i++) {
		MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		MPI_Send(range, 2, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
	}
}
#endif

#ifdef WITH_MPI
static void
slave(context *context)
{
	int range[2];

	for (;;) {
		MPI_Send(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(range, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		if (range[0] == -1 ||
		    range[1] == -1)
			break;

		while (task_n < range[0]) {
			task_next();
			task_n++;
		}

		while (this_n < range[1]) {
			task_exec();
			task_next();
			task_n++;
		}
	}
}
#endif

void
queue_balance_work(context *context)
{
#ifdef WITH_MPI
	int rank, size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size == 1)
		do_all_tasks(context);
	else {
		MPI_Barrier(MPI_COMM_WORLD);

		if (rank == 0)
			master(size);
		else
			slave(context);

		MPI_Barrier(MPI_COMM_WORLD);
	}
#else
	do_all_tasks(context);
#endif
}
//XXX mpi
#endif

