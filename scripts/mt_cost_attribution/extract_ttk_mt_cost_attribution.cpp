#include <MergeTreeDistance.h>
#include <ttkMergeTreeUtils.h>

#include <vtkNew.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

class AuditableMergeTreeDistance : public ttk::MergeTreeDistance {
public:
  template <class dataType>
  void preprocessExact(ttk::ftm::MergeTree<dataType> &tree,
                       std::vector<int> &nodeCorr) {
    // Audited defaults used by the production DistanceMatrix path:
    // epsilon1=0, epsilon2=100, epsilon3=100,
    // branch decomposition=true, use min-max pair=true, clean tree=true.
    this->preprocessingPipeline<dataType>(
      tree, 0.0, 100.0, 100.0, true, true, true, nodeCorr);
  }

  template <class dataType>
  dataType deleteCostPublic(const ttk::ftm::FTMTree_MT *tree,
                            ttk::ftm::idNode nodeId) {
    return this->deleteCost<dataType>(tree, nodeId);
  }

  template <class dataType>
  dataType insertCostPublic(const ttk::ftm::FTMTree_MT *tree,
                            ttk::ftm::idNode nodeId) {
    return this->insertCost<dataType>(tree, nodeId);
  }
};

struct Args {
  std::string nodes1;
  std::string arcs1;
  std::string nodes2;
  std::string arcs2;
  std::string summaryCsv;
  std::string unmatchedCsv;
  std::string matchingCsv;
};

void usage(const char *argv0) {
  std::cerr
    << "Usage:\n  " << argv0
    << " --nodes1 A_nodes.vtu --arcs1 A_arcs.vtu"
       " --nodes2 B_nodes.vtu --arcs2 B_arcs.vtu"
       " --summary-csv summary.csv"
       " --unmatched-csv unmatched.csv"
       " --matching-csv matching.csv\n";
}

Args parseArgs(int argc, char **argv) {
  Args a;

  auto requireValue = [&](int &i) -> std::string {
    if(i + 1 >= argc)
      throw std::runtime_error("Missing value after " + std::string(argv[i]));
    return std::string(argv[++i]);
  };

  for(int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if(key == "--nodes1")
      a.nodes1 = requireValue(i);
    else if(key == "--arcs1")
      a.arcs1 = requireValue(i);
    else if(key == "--nodes2")
      a.nodes2 = requireValue(i);
    else if(key == "--arcs2")
      a.arcs2 = requireValue(i);
    else if(key == "--summary-csv")
      a.summaryCsv = requireValue(i);
    else if(key == "--unmatched-csv")
      a.unmatchedCsv = requireValue(i);
    else if(key == "--matching-csv")
      a.matchingCsv = requireValue(i);
    else if(key == "-h" || key == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else
      throw std::runtime_error("Unknown argument: " + key);
  }

  for(const auto *p :
      {&a.nodes1, &a.arcs1, &a.nodes2, &a.arcs2,
       &a.summaryCsv, &a.unmatchedCsv, &a.matchingCsv}) {
    if(p->empty())
      throw std::runtime_error("Missing required argument.");
  }

  return a;
}

vtkSmartPointer<vtkUnstructuredGrid> readGrid(const std::string &path) {
  vtkNew<vtkXMLUnstructuredGridReader> r;
  r->SetFileName(path.c_str());
  r->Update();

  auto *raw = r->GetOutput();
  if(raw == nullptr || raw->GetNumberOfPoints() <= 0)
    throw std::runtime_error("Failed to read VTU: " + path);

  vtkSmartPointer<vtkUnstructuredGrid> out
    = vtkSmartPointer<vtkUnstructuredGrid>::New();
  out->ShallowCopy(raw);
  return out;
}

struct SideStats {
  std::size_t activeNodes{0};
  std::size_t matchedNodes{0};
  std::size_t unmatchedNodes{0};
  double nonmatchingCost{0.0};
};

template <class dataType>
void writeUnmatchedSide(
  std::ofstream &out,
  const char *side,
  ttk::ftm::FTMTree_MT *tree,
  const std::set<ttk::ftm::idNode> &matched,
  AuditableMergeTreeDistance &distance,
  SideStats &stats) {

  stats = SideStats{};

  const auto n = tree->getNumberOfNodes();

  for(ttk::ftm::idNode node = 0; node < n; ++node) {
    if(tree->isNodeAlone(node))
      continue;

    ++stats.activeNodes;

    if(matched.count(node)) {
      ++stats.matchedNodes;
      continue;
    }

    ++stats.unmatchedNodes;

    const double cost = static_cast<double>(
      side[0] == '1'
        ? distance.deleteCostPublic<dataType>(tree, node)
        : distance.insertCostPublic<dataType>(tree, node));

    stats.nonmatchingCost += cost;

    const auto bd = tree->getBirthDeath<dataType>(node);
    const double birth = static_cast<double>(std::get<0>(bd));
    const double death = static_cast<double>(std::get<1>(bd));
    const double persistence = std::abs(death - birth);

    const auto origin = tree->getNode(node)->getOrigin();

    out << side << ","
        << static_cast<long long>(node) << ","
        << static_cast<long long>(origin) << ","
        << (tree->isRoot(node) ? 1 : 0) << ","
        << birth << ","
        << death << ","
        << persistence << ","
        << cost << "\n";
  }
}

template <class dataType>
int run(const Args &args) {
  auto nodes1 = readGrid(args.nodes1);
  auto arcs1 = readGrid(args.arcs1);
  auto nodes2 = readGrid(args.nodes2);
  auto arcs2 = readGrid(args.arcs2);

  auto tree1 = ttk::ftm::makeTree<dataType>(nodes1, arcs1);
  auto tree2 = ttk::ftm::makeTree<dataType>(nodes2, arcs2);

  AuditableMergeTreeDistance distance;
  distance.setThreadNumber(1);
  distance.setCleanTree(true);
  distance.setIsCalled(true);

  // Explicitly reproduce TTK preprocessing so that the exact internal
  // branch-decomposition trees remain available for node-level attribution.
  std::vector<int> corr1, corr2;
  distance.preprocessExact<dataType>(tree1, corr1);
  distance.preprocessExact<dataType>(tree2, corr2);

  // Execute on the already-preprocessed trees. No second preprocessing and
  // no postprocessing/conversion are allowed here.
  distance.setPreprocess(false);
  distance.setPostprocess(false);
  distance.setSaveTree(false);

  std::vector<
    std::tuple<ttk::ftm::idNode, ttk::ftm::idNode, double>
  > matching;

  const double d = static_cast<double>(
    distance.execute<dataType>(tree1, tree2, matching));

  std::set<ttk::ftm::idNode> matched1;
  std::set<ttk::ftm::idNode> matched2;

  double relabelSum = 0.0;

  std::ofstream mout(args.matchingCsv);
  if(!mout)
    throw std::runtime_error("Cannot open matching CSV: " + args.matchingCsv);

  mout << std::setprecision(17);
  mout << "tree1_node_id,tree2_node_id,relabel_cost\n";

  for(const auto &m : matching) {
    const auto n1 = std::get<0>(m);
    const auto n2 = std::get<1>(m);
    const double c = std::get<2>(m);

    if(!matched1.insert(n1).second)
      throw std::runtime_error("tree1 node appears in multiple raw matches");
    if(!matched2.insert(n2).second)
      throw std::runtime_error("tree2 node appears in multiple raw matches");

    relabelSum += c;

    mout << static_cast<long long>(n1) << ","
         << static_cast<long long>(n2) << ","
         << c << "\n";
  }

  std::ofstream uout(args.unmatchedCsv);
  if(!uout)
    throw std::runtime_error("Cannot open unmatched CSV: " + args.unmatchedCsv);

  uout << std::setprecision(17);
  uout
    << "side,node_id,origin_id,is_root,"
       "birth_raw,death_raw,persistence_raw,nonmatching_cost\n";

  SideStats s1, s2;

  writeUnmatchedSide<dataType>(
    uout, "1_delete", &tree1.tree, matched1, distance, s1);

  writeUnmatchedSide<dataType>(
    uout, "2_insert", &tree2.tree, matched2, distance, s2);

  const double totalSq = d * d;
  const double recomposed = relabelSum + s1.nonmatchingCost + s2.nonmatchingCost;
  const double residual = totalSq - recomposed;

  std::ofstream sout(args.summaryCsv);
  if(!sout)
    throw std::runtime_error("Cannot open summary CSV: " + args.summaryCsv);

  sout << std::setprecision(17);
  sout
    << "distance,distance_squared,"
       "raw_matching_count,relabel_cost_sum,"
       "tree1_active_nodes,tree1_matched_nodes,tree1_unmatched_nodes,"
       "tree1_delete_cost_sum,"
       "tree2_active_nodes,tree2_matched_nodes,tree2_unmatched_nodes,"
       "tree2_insert_cost_sum,"
       "recomposed_squared,recomposition_residual\n";

  sout
    << d << ","
    << totalSq << ","
    << matching.size() << ","
    << relabelSum << ","
    << s1.activeNodes << ","
    << s1.matchedNodes << ","
    << s1.unmatchedNodes << ","
    << s1.nonmatchingCost << ","
    << s2.activeNodes << ","
    << s2.matchedNodes << ","
    << s2.unmatchedNodes << ","
    << s2.nonmatchingCost << ","
    << recomposed << ","
    << residual << "\n";

  std::cout << std::setprecision(17);
  std::cout << "distance=" << d << "\n";
  std::cout << "distance_squared=" << totalSq << "\n";
  std::cout << "raw_matching_count=" << matching.size() << "\n";
  std::cout << "relabel_cost_sum=" << relabelSum << "\n";
  std::cout << "tree1_active_nodes=" << s1.activeNodes << "\n";
  std::cout << "tree1_unmatched_nodes=" << s1.unmatchedNodes << "\n";
  std::cout << "tree1_delete_cost_sum=" << s1.nonmatchingCost << "\n";
  std::cout << "tree2_active_nodes=" << s2.activeNodes << "\n";
  std::cout << "tree2_unmatched_nodes=" << s2.unmatchedNodes << "\n";
  std::cout << "tree2_insert_cost_sum=" << s2.nonmatchingCost << "\n";
  std::cout << "recomposed_squared=" << recomposed << "\n";
  std::cout << "recomposition_residual=" << residual << "\n";

  // Acceptance criterion: manual preprocessing + explicit unmatched-node
  // attribution must reproduce the exact DP objective within float-level
  // tolerance. If this fails, individual delete/insert attribution is rejected.
  const double tol = 1e-5 * std::max(1.0, totalSq);

  if(std::abs(residual) > tol) {
    std::cerr
      << "FAIL: unmatched-node delete/insert attribution does not "
         "recompose the TTK objective within tolerance.\n";
    return 20;
  }

  std::cout << "NODEWISE_DELETE_INSERT_RECOMPOSITION=PASS\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto args = parseArgs(argc, argv);
    return run<float>(args);
  } catch(const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
