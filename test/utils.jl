using Artifacts

"""
Helper function to obtain the filepaths for an instance's model, evidence, and
solution files within the uai2014 artifact, corresponding to the provided
task.

# Arguments
- `problem_name::String`: The name of the problem or instance.
- `task::String`: The task type. Supported tasks are "MAR", "MPE", "MMAP", and "PR".
"""
function get_instance_filepaths(problem_name::AbstractString, task::AbstractString)
  model_filepath = joinpath(artifact"uai2014", task, problem_name * ".uai")
  evid_filepath = joinpath(artifact"uai2014", task, problem_name * ".uai.evid")
  sol_filepath = joinpath(artifact"uai2014", task, problem_name * ".uai." * task)
  return model_filepath, evid_filepath, sol_filepath
end
