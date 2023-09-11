# Color palettes: 
# - https://github.com/matplotlib/matplotlib/issues/9460#issuecomment-875185352
# - https://personal.sron.nl/~pault/#tab:blindvision
# - https://yoshke.org/blog/colorblind-friendly-diagrams
# - https://davidmathlogic.com/colorblind
# - https://www.color-hex.com/color-palette/1018347

using PGFPlotsX, CSV, DataFrames, Artifacts, StatsBase
using JunctionTrees: get_td_soln

push!(
  PGFPlotsX.CUSTOM_PREAMBLE,
  """
\\usepackage[T1]{fontenc}
\\usepackage{xcolor}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\usepackage{amsmath,amssymb}
% Bright qualitative colour scheme that is colour-blind safe
% https://personal.sron.nl/~pault/#tab:blindvision
\\definecolor{c01}{HTML}{4477AA}
\\definecolor{c02}{HTML}{EE6677}
\\definecolor{c03}{HTML}{228833}
\\definecolor{c04}{HTML}{CCBB44}
\\definecolor{c05}{HTML}{66CCEE}
\\definecolor{c06}{HTML}{AA3377}
\\definecolor{c07}{HTML}{BBBBBB}
\\definecolor{c08}{HTML}{BBBBBB}
\\usepackage{fontsetup}
\\setmonofont{Hack}
"""
)

# data_file = ARGS[1]

# DEBUG
data_file = "./out/co23/2023-09-10--20-20-45/out.csv"

df1 =
  CSV.File(data_file) |> # read benchmark data from file
  DataFrame |> # convert it to a data_file frame
   x -> unstack(x, :problem, :library, :execution_time) |> # convert from long to wide format (create a column for each possible `library` value)
   dropmissing # drop rows with missing values

df2 =
  map(x -> joinpath(artifact"uai2014-mar", x * ".tamaki.td"), df1.problem) |> # create absolute filepaths for each problem in `df`
  x -> get_td_soln.(x) |> # get the tree decomposition solution line ([nbags, :largest_bag_size, nvars]) for each problem 
  x -> DataFrame(x=[i for i in x]) |> # create data frame using the constructor for vector of vectors
  x -> select(x, :x => AsTable) |> # https://www.juliabloggers.com/handling-vectors-of-vectors-in-dataframes-jl/
  x -> rename(x, [:nbags, :largest_bag_size, :nvars]) # rename columns

df =
  hcat(df1, df2) |> # horizontally concat data frame
  x -> sort(x, [:largest_bag_size, :nvars, :nbags]) |> # sort the rows based on the largest bag size
  x -> transform(x, [:libdai, :TensorInference] => (./) => :libdai_ti_speedup) |>
  x -> transform(x, [:merlin, :TensorInference] => (./) => :merlin_ti_speedup) |>
  x -> transform(x, [:JunctionTrees_v1, :TensorInference] => (./) => :jtv1_ti_speedup) |>
  x -> transform(x, [:JunctionTrees_v2, :TensorInference] => (./) => :jtv2_ti_speedup)

labels =
  df.problem |>
  x -> match.(r"[a-zA-Z]+", x) |>
  x -> getfield.(x, :match)

labels_unique = unique(labels) |> sort
xmax = maximum(df.largest_bag_size) + 1

@pgf tp = Axis(
  {
    # title="TensorInference.jl Speedup",
    xmin = 0,
    xmax = xmax,
    xlabel = "Largest cluster size",
    xmajorgrids = true,
    ymin = 0,
    ymax = 1000000,
    ymode = "log",
    ytick = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
    ymajorgrids = true,
    ylabel = "Run time speedup",
    label_style = {font = raw"\footnotesize"},
    tick_label_style = {font = raw"\footnotesize"},
    "scatter/classes" = {
      # Warning: These classes be defined in sorted to keep the correspondance with `labels_unique`
      Alchemy = {
        mark = "x",
      },
      CSP = {
        mark = "+",
      },
      # DBN = {
      #   mark = "square"
      # },
      Grids = {
        mark = "asterisk",
      },
      ObjectDetection = {
        mark = "-",
      },
      Pedigree = {
        mark = "triangle",
      },
      Promedus = {
        mark = "o",
        # mark = "square",
        # mark = "pentagon",
      },
      Segmentation = {
        mark = "Mercedes star",
      },
      linkage = {
        mark = "diamond",
        # mark = "|",
      },
    },
    legend_style = {
      legend_columns = 3,
      at = Coordinate(0.51, -0.4),
      anchor = "south",
      draw = "none",
      font = raw"\footnotesize",
      column_sep = 1.5,
    },
  },
  Plot(
    {
      c01,
      scatter,
      "only marks",
      "scatter src" = "explicit symbolic",
      "legend image post style" = "black", "legend style" = {text = "black", font = raw"\footnotesize"},
    },
    Table(
      {
        meta = "label"
      },
      x=df.largest_bag_size,
      y=df.libdai_ti_speedup,
      label=labels,
    ),
  ),
  Plot(
    {
      c02,
      scatter,
      "only marks",
      "scatter src" = "explicit symbolic",
    },
    Table(
      {
        meta = "label"
      },
      x=df.largest_bag_size,
      y=df.merlin_ti_speedup,
      label=labels,
    ),
  ),
  Plot(
    {
      c03,
      scatter,
      "only marks",
      "scatter src" = "explicit symbolic",
    },
    Table(
      {
        meta = "label"
      },
      x=df.largest_bag_size,
      y=df.jtv1_ti_speedup,
      label=labels,
    ),
  ),
  Plot(
    {
      c04,
      scatter,
      "only marks",
      "scatter src" = "explicit symbolic",
    },
    Table(
      {
        meta = "label"
      },
      x=df.largest_bag_size,
      y=df.jtv2_ti_speedup,
      label=labels,
    ),
  ),
  HLine({ dashed, black }, 1), # See: https://kristofferc.github.io/PGFPlotsX.jl/v1/examples/convenience/
  Legend(labels_unique),
  # Library legend (manually made with LaTeX code. See: https://kristofferc.github.io/PGFPlotsX.jl/v1/examples/latex/)
  [raw"\node ",
    {
      draw = "black",
      fill = "white",
      font = raw"\scriptsize",
      # pin = "outlier"
    },
    " at ",
    Coordinate(5.5, 30000), # warning: hardcoded!
    raw"{\shortstack[l] { $\textcolor{c01}{\blacksquare}$ libDAI \\ $\textcolor{c02}{\blacksquare}$ Merlin \\ $\textcolor{c03}{\blacksquare}$ JunctionTrees.jl-v1  \\ $\textcolor{c04}{\blacksquare}$ JunctionTrees.jl-v2}};"
  ]
)

println("Geometric mean of the speedup: $(geomean(vcat(df.libdai_ti_speedup, df.merlin_ti_speedup, df.jtv1_ti_speedup, df.jtv2_ti_speedup)))")
println("Geometric mean of the speedup of the last 10 problems: $(geomean(vcat(last(df.libdai_ti_speedup, 10), last(df.merlin_ti_speedup, 10), last(df.jtv1_ti_speedup, 10), last(df.jtv2_ti_speedup, 10))))")

output_file = joinpath(dirname(data_file), "performance-evaluation.svg")
pgfsave(output_file, tp; include_preamble=true, dpi=150)

# DEBUG
display(tp)
