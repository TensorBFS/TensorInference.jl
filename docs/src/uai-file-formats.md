# UAI file formats

The UAI Format consists of four potential parts (each associated with a file):

- [Model file format (.uai)](@ref).
- [Evidence file format (.evid)](@ref).
- [Query file format (.query)](@ref).
- [Results file format (.MAR, .MAP, .MMAP .PR)](@ref).

## Model file format (.uai)

We use the simple text file format specified below to describe problem
instances (Markov networks). The format is a generalization of the Ergo file
format initially developed by Noetic Systems Inc. for their Ergo software. We
use the *.uai* suffix for the evaluation benchmark network files.

### Structure

A file in the UAI format consists of the following two parts, in that order:

    <Preamble>

    <Function tables>

The contents of each section (denoted `<...>` above) are described in the
following:

#### Preamble

Our description of the format will follow a simple Markov network with three
variables and two functions. A sample preamble for such a network is:

    MARKOV
    3
    2 2 3
    2
    2 0 1
    2 1 2

The preamble starts with one line denoting the type of network. Generally, this
can be either `BAYES` (if the network is a Bayesian network) or `MARKOV` (in
case of a Markov network). However, note that this year all networks will be
given in a Markov networks (i.e. Bayesian networks will be moralized).

The second line contains the number of variables. The next line specifies the
cardinalities of each variable, one at a time, separated by a whitespace (note
that this implies an order on the variables which will be used throughout the
file). The fourth line contains only one integer, denoting the number of
cliques in the problem. Then, one clique per line, the scope of each clique is
given as follows: The first integer in each line specifies the number of
variables in the clique, followed by the actual indexes of the variables. The
order of this list is not restricted. Note that the ordering of variables
within a factor will follow the order provided here.

Referring to the example above, the first line denotes the Markov network, the
second line tells us the problem consists of three variables, let's refer to
them as `X`, `Y`, and `Z`. Their cardinalities are 2, 2, and 3 respectively
(from the third line). Line four specifies that there are 2 cliques. The first
clique is `X,Y`, while the second clique is `Y,Z`. Note that variables are
indexed starting with 0.

#### Function tables 

In this section each factor is specified by giving its full table (i.e,
specifying value for each assignment). The order of the factor is identical to
the one in which they were introduced in the preamble, the first variable have
the role of the 'most significant' digit. For each factor table, first the
number of entries is given (this should be equal to the product of the domain
sizes of the variables in the scope). Then, one by one, separated by
whitespace, the values for each assignment to the variables in the function's
scope are enumerated. Tuples are implicitly assumed in ascending order, with
the last variable in the scope as the 'least significant'. To illustrate, we
continue with our Markov network example from above, let's assume the following
conditional probability tables:

    X     P(X)
    0     0.436
    1     0.564

    X Y   P(Y,X)
    0 0   0.128
    0 1   0.872
    1 0   0.920
    1 1   0.080

    Y Z   P(Z,Y)
    0 0   0.210
    0 1   0.333
    0 2   0.457
    1 0   0.811
    1 1   0.000
    1 2   0.189

The corresponding function tables in the file would then look like this:

    2
    0.436 0.564

    4
    0.128 0.872
    0.920 0.080

    6
    0.210 0.333 0.457
    0.811 0.000 0.189

(Note that line breaks and empty lines are effectively just a whitespace,
exactly like plain spaces " ". They are used here to improve readability.)

### Summary

To sum up, a problem file consists of 2 sections: the preamble and the full the
function tables, the names and the labels. For our Markov network example
above, the full file will look like:

    MARKOV
    3
    2 2 3
    3
    1 0
    2 0 1
    2 1 2

    2
    0.436 0.564

    4
    0.128 0.872
    0.920 0.080

    6
    0.210 0.333 0.457
    0.811 0.000 0.189 

## Evidence file format (.evid)

Evidence is specified in a separate file. This file has the same name as the
original network file but with an added *.evid* suffix. For instance,
*problem.uai* will have evidence in *problem.uai.evid*. The file starts with a
line specifying the number of evidences samples. The evidence in each sample,
will be written in a new line. Each line will begin with the number of observed
variables in the sample, followed by pairs of variable and its observed value.
The indexes correspond to the ones implied by the original problem file. If,
for our above example, we want to provide a single sample where the variable Y
  has been observed as having its first value and Z with its second value, the
  file *example.uai.evid* would contain the following:

    1
    2 1 0 2 1

## Query file format (.query)

Query variables for marginal MAP (MMAP) inference are specified in a separate
file. This file has the same name as the original network file but with an added
*.query* suffix. For instance with respect to the UAI model format,
*problem.uai* will have evidence in *problem.uai.query*.

The query file consists of a single line. The line will begin with the number of
query variables, followed by the indexes of the query variables. The indexes
correspond to the ones implied by the original problem file.

For our example Markov network given Model Format, if we wanted to use Y as the
query variable the file *example.uai.query* would contain the following:

    1 1

As a second example, if variables with indices 0, 4, 8 and 17 are query
variables, the query file would contain the following:

    4 0 4 8 17

## Results file format (.MAR, .MAP, .MMAP .PR)

The first line must contain only the task solved: `MAR|MAP|MAPP|PR`. The rest of
the file will contain the solution for the task. Solvers can write more then one
solution by writing `-BEGIN-` at the head of the new solution. We will only
consider the last solution in the file. In the example below the task we choose
is `PR`. We have two solutions. The format of the `<SOLUTION>` part will be
described below.

    PR
    <SOLUTION>
    -BEGIN-
    <SOLUTION>

The first line in each solution will contain the number of evidence samples.
This will be the number of lines (not include this line) in the solution part.
Hence each line from here will contain the solution with a different sample of
evidence - ordered as in the evidence file. If there is no evidence (the first
line of the evidence file is 0), the output should include the results for the
empty evidence scenario. This is regarded as a single-evidence case - one with
the empty evidence.

Solvers that can bound their estimation are encouraged to specify if their
solution is lower or upper bound. Doing so by adding at the end of the solution
the letters `L` (for lower bound) or `U` (for upper bound).

The line format is as follows (according to the task):

- **Partition function, PR**: Line with the value of the log10 of the partition
  function. For example, an approximation log10 Pr(e) = -0.2008 which is known
  to be an upper bound may have a solution line:

      -0.2008 U

- **Most probable explanation, MAP**: A space separated line that includes:
  1. the number n of model variables, and
  2. the `MAP` instantiation, a list of value indices for all n variables.
  For example, an input model with 3 binary variables may have a solution line:

      3 0 1 0

- **Marginals, MAR**: A space separated line that includes:
  1. The number of variables in the model.
  2. A list of marginal approximations of all the variables. For each variable
     its cardinality is first stated, then the probability of each state is
     stated. The order of the variables is the same as in the model, all data
     is space separated.

  For example, a model with 3 variables, with cardinalities of 2, 2, 3
  respectively. The solution might look like this:

      3 2 0.1 0.9 2 0.3 0.7 3 0.2 0.2 0.6

- **Beliefs, BEL**: A space separated line that includes:

  1. The number n of model cliques, and

  2. A list of belief approximations for all n cliques. Each marginal
     approximation is specified by a list, starting with the number of entries
     of the factor, followed by the approximation ``Pr(x|e)`` for each value of
     ``x`` (where is a vector of the clique variables).

  For example, if an input model has 2 cliques the first with 2 binary variable
  and the second with 3. The solution line may look like:

      2 4 0.25 0.25 0.4 0.1 8 0.1 0.05 0.05 0.2 0.1 0.01 0.04 0.45

  The order of the entries is as in the model description.

Here is a complete example for a solution for the `MAP` task. The evidence file
contains one evidence samples.

    MAP
    1
    4 0 2 0 4
    -BEGIN-
    1
    4 0 2 0 4

If a solver does not produce a solution by the given time, it would be
considered as having failed on the instance. This will be treated as equivalent
to a naive solution (e.g. bit-wise singleton clique maximum for a `MAP`
problem).
