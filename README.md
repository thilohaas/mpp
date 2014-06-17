mpp
===

Assignments for Mainframe &amp; Parallel Programming lecture

Setup
=====

Assignment 1 was implemented using the message passing scheme with open-mpi and the unofficial python library mpi4py which provides bindings for the Message Passing Interface standard for python. The histogram graphs are created using the pyplot library.

Assignment 2 uses the shared address paradigm and is implemented using python’s multiprocessing package and numpy for better and faster array handling.

To install all required dependencies run the following commands within the projects directory:

```
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

