#include <MergeTreeDistance.h>
#include <ttkMergeTreeUtils.h>

#include <vtkNew.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Args {
  std::string nodes1;
  std::string arcs1;
  std::string nodes2;
  std::string arcs2;
  std::string matchingCsv;
  std::string summaryCsv;
  bool postprocess{false};
};

void usage(const char *argv0) {
  std::cerr
    << "Usage:\n  " << argv0
    << " --nodes1 A.vtu --arcs1 A.vtu"
       " --nodes2 B.vtu --arcs2 B.vtu"
       " --postprocess 0|1"
       " --matching-csv out.csv"
       " --summary-csv summary.csv\n";
}

Args parseArgs(int argc, char **argv) {
  Args a;

  auto requireValue = [&](int &i) -> std::string {
    if(i + 1 >= argc) {
      throw std::runtime_error("Missing value after " + std::string(argv[i]));
    }
    return std::string(argv[++i]);
  };

  for(int i = 1; i < argc; ++i) {
    const std::string key = argv[i];

    if(key == "--nodes1") {
      a.nodes1 = requireValue(i);
    } else if(key == "--arcs1") {
      a.arcs1 = requireValue(i);
    } else if(key == "--nodes2") {
      a.nodes2 = requireValue(i);
    } else if(key == "--arcs2") {
      a.arcs2 = requireValue(i);
    } else if(key == "--matching-csv") {
      a.matchingCsv = requireValue(i);
    } else if(key == "--summary-csv") {
      a.summaryCsv = requireValue(i);
    } else if(key == "--postprocess") {
      const auto v = requireValue(i);
      if(v == "0") {
        a.postprocess = false;
      } else if(v == "1") {
        a.postprocess = true;
      } else {
        throw std::runtime_error("--postprocess must be 0 or 1");
      }
    } else if(key == "-h" || key == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }

  for(const auto *p : {
        &a.nodes1, &a.arcs1, &a.nodes2, &a.arcs2,
        &a.matchingCsv, &a.summaryCsv}) {
    if(p->empty()) {
      throw std::runtime_error("Missing required argument.");
    }
  }

  return a;
}

vtkSmartPointer<vtkUnstructuredGrid> readGrid(const std::string &path) {
  vtkNew<vtkXMLUnstructuredGridReader> r;
  r->SetFileName(path.c_str());
  r->Update();

  auto *raw = r->GetOutput();
  if(raw == nullptr) {
    throw std::runtime_error("Failed to read VTU: " + path);
  }

  vtkSmartPointer<vtkUnstructuredGrid> out
    = vtkSmartPointer<vtkUnstructuredGrid>::New();
  out->ShallowCopy(raw);
  return out;
}

template <class dataType>
double runOnce(
  vtkUnstructuredGrid *nodes1,
  vtkUnstructuredGrid *arcs1,
  vtkUnstructuredGrid *nodes2,
  vtkUnstructuredGrid *arcs2,
  const bool postprocess,
  std::vector<std::tuple<ttk::ftm::idNode, ttk::ftm::idNode, double>>
    &matching) {

  // This is the same VTK -> internal-tree construction path used by TTK's
  // merge-tree utilities for ordinary node/arc multiblock inputs.
  auto tree1 = ttk::ftm::makeTree<dataType>(nodes1, arcs1);
  auto tree2 = ttk::ftm::makeTree<dataType>(nodes2, arcs2);

  ttk::MergeTreeDistance distance;

  // Reproduce the pairwise DistanceMatrix execution behavior.
  // Algorithmic parameters remain at the audited TTK 1.3.0 defaults.
  distance.setThreadNumber(1);
  distance.setSaveTree(true);
  distance.setCleanTree(true);
  distance.setIsCalled(true);
  distance.setPostprocess(postprocess);

  matching.clear();

  return static_cast<double>(
    distance.execute<dataType>(tree1, tree2, matching));
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto args = parseArgs(argc, argv);

    auto nodes1 = readGrid(args.nodes1);
    auto arcs1 = readGrid(args.arcs1);
    auto nodes2 = readGrid(args.nodes2);
    auto arcs2 = readGrid(args.arcs2);

    std::vector<
      std::tuple<ttk::ftm::idNode, ttk::ftm::idNode, double>
    > matching;

    const double distance = runOnce<float>(
      nodes1,
      arcs1,
      nodes2,
      arcs2,
      args.postprocess,
      matching);

    std::ofstream mout(args.matchingCsv);
    if(!mout) {
      throw std::runtime_error(
        "Cannot open matching CSV: " + args.matchingCsv);
    }

    mout << "tree1_node_id,tree2_node_id,relabel_cost\n";
    mout << std::setprecision(17);

    long long max1 = -1;
    long long max2 = -1;

    for(const auto &m : matching) {
      const auto n1 = static_cast<long long>(std::get<0>(m));
      const auto n2 = static_cast<long long>(std::get<1>(m));
      const double c = std::get<2>(m);

      mout << n1 << "," << n2 << "," << c << "\n";

      max1 = std::max(max1, n1);
      max2 = std::max(max2, n2);
    }

    std::ofstream sout(args.summaryCsv);
    if(!sout) {
      throw std::runtime_error(
        "Cannot open summary CSV: " + args.summaryCsv);
    }

    sout
      << "postprocess,distance,matching_count,"
         "input_tree1_nodes,input_tree2_nodes,"
         "max_tree1_match_id,max_tree2_match_id\n";

    sout << std::setprecision(17)
         << (args.postprocess ? 1 : 0) << ","
         << distance << ","
         << matching.size() << ","
         << nodes1->GetNumberOfPoints() << ","
         << nodes2->GetNumberOfPoints() << ","
         << max1 << ","
         << max2 << "\n";

    std::cout << std::setprecision(17);
    std::cout << "postprocess=" << (args.postprocess ? 1 : 0) << "\n";
    std::cout << "distance=" << distance << "\n";
    std::cout << "matching_count=" << matching.size() << "\n";
    std::cout << "input_tree1_nodes=" << nodes1->GetNumberOfPoints() << "\n";
    std::cout << "input_tree2_nodes=" << nodes2->GetNumberOfPoints() << "\n";
    std::cout << "max_tree1_match_id=" << max1 << "\n";
    std::cout << "max_tree2_match_id=" << max2 << "\n";

    return 0;

  } catch(const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
