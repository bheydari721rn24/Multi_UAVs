"""
Topological Data Analysis Module for Multi-UAV Systems

This module implements topological methods for analyzing spatial patterns and formations,
using persistent homology to detect and characterize emergent structures.

Author: Research Team
Date: 2025
"""

import numpy as np
import math
import networkx as nx
from utils import CONFIG


class TopologyAnalyzer:
    """
    Topological data analysis for pattern recognition in swarm formations
    Uses persistent homology to identify significant features in spatial data
    """
    
    def __init__(self):
        """Initialize topology analyzer with configuration parameters"""
        self.config = CONFIG["swarm"]["topology"]
        self.persistence_threshold = self.config["persistence_threshold"]
        self.filtration_steps = self.config["filtration_steps"]
        self.homology_dimension_max = self.config["homology_dimension_max"]
        self.significance_threshold = self.config["feature_significance_threshold"]
        self.update_rate = self.config["pattern_recognition_update_rate"]
        
        # Last computed persistence diagram
        self.last_persistence_diagram = None
        
        # Detected formation patterns
        self.detected_patterns = None
        
        # Feature history
        self.feature_history = []
        self.max_history_length = 10
        
    def compute_distance_matrix(self, point_cloud):
        """
        Compute pairwise distance matrix for point cloud
        
        Args:
            point_cloud: List of points (UAV positions)
            
        Returns:
            numpy.ndarray: Distance matrix
        """
        n_points = len(point_cloud)
        distance_matrix = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.linalg.norm(point_cloud[i] - point_cloud[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                
        return distance_matrix
        
    def compute_vietoris_rips_complex(self, distance_matrix, epsilon):
        """
        Compute Vietoris-Rips complex for a given distance threshold
        
        Args:
            distance_matrix: Pairwise distance matrix
            epsilon: Distance threshold for complex
            
        Returns:
            list: List of simplices in the complex
        """
        n_points = distance_matrix.shape[0]
        simplices = []
        
        # Add 0-simplices (vertices)
        for i in range(n_points):
            simplices.append([i])
            
        # Add 1-simplices (edges)
        for i in range(n_points):
            for j in range(i+1, n_points):
                if distance_matrix[i, j] <= epsilon:
                    simplices.append([i, j])
                    
        # Add 2-simplices (triangles)
        for i in range(n_points):
            for j in range(i+1, n_points):
                for k in range(j+1, n_points):
                    if (distance_matrix[i, j] <= epsilon and
                        distance_matrix[i, k] <= epsilon and
                        distance_matrix[j, k] <= epsilon):
                        simplices.append([i, j, k])
                        
        # Add 3-simplices (tetrahedra) - only if needed
        if self.homology_dimension_max >= 3:
            for i in range(n_points):
                for j in range(i+1, n_points):
                    for k in range(j+1, n_points):
                        for l in range(k+1, n_points):
                            if (distance_matrix[i, j] <= epsilon and
                                distance_matrix[i, k] <= epsilon and
                                distance_matrix[i, l] <= epsilon and
                                distance_matrix[j, k] <= epsilon and
                                distance_matrix[j, l] <= epsilon and
                                distance_matrix[k, l] <= epsilon):
                                simplices.append([i, j, k, l])
                                
        return simplices
        
    def compute_betti_numbers(self, simplices):
        """
        Compute Betti numbers from simplicial complex
        
        Args:
            simplices: List of simplices
            
        Returns:
            list: Betti numbers [β₀, β₁, β₂]
        """
        # This is a simplified implementation of computing Betti numbers
        # In a full implementation, we would use a proper library for this
        
        # Group simplices by dimension
        simplices_by_dim = {}
        for simplex in simplices:
            dim = len(simplex) - 1
            if dim not in simplices_by_dim:
                simplices_by_dim[dim] = []
            simplices_by_dim[dim].append(simplex)
            
        # Betti numbers to compute
        betti = [0] * (self.homology_dimension_max + 1)
        
        # Calculate β₀ (number of connected components)
        if 0 in simplices_by_dim:
            # Create a graph from 1-simplices (edges)
            G = nx.Graph()
            
            # Add all vertices
            for simplex in simplices_by_dim[0]:
                G.add_node(simplex[0])
                
            # Add all edges
            if 1 in simplices_by_dim:
                for simplex in simplices_by_dim[1]:
                    G.add_edge(simplex[0], simplex[1])
                    
            # Number of connected components is β₀
            betti[0] = nx.number_connected_components(G)
        
        # Calculate β₁ (number of 1-dimensional holes)
        if 1 in simplices_by_dim:
            # Create a graph from 1-simplices (edges)
            G = nx.Graph()
            
            # Add all edges
            for simplex in simplices_by_dim[1]:
                G.add_edge(simplex[0], simplex[1])
                
            # Fill in triangles (2-simplices)
            triangles_filled = 0
            if 2 in simplices_by_dim:
                triangles_filled = len(simplices_by_dim[2])
                
            # β₁ = number of cycles - number of filled triangles
            # Number of cycles = |E| - |V| + number of connected components
            # where |E| = number of edges, |V| = number of vertices
            num_edges = G.number_of_edges()
            num_vertices = G.number_of_nodes()
            num_components = nx.number_connected_components(G)
            
            cycles = num_edges - num_vertices + num_components
            betti[1] = max(0, cycles - triangles_filled)
            
        # Higher Betti numbers would require more sophisticated calculations
        # For simplicity, we'll use a placeholder value for β₂
        if 2 in simplices_by_dim and 3 in simplices_by_dim:
            # Rough approximation - more accurate calculation would use boundary matrices
            betti[2] = max(0, len(simplices_by_dim[2]) - len(simplices_by_dim[3]))
            
        return betti
        
    def compute_persistence_diagram(self, point_cloud):
        """
        Compute topological features using persistent homology
        
        Args:
            point_cloud: List of points (UAV positions)
            
        Returns:
            list: Persistence diagram as list of (birth, death, dimension) triples
        """
        # Compute distance matrix
        distance_matrix = self.compute_distance_matrix(point_cloud)
        
        # Sort unique distance values to use as filtration thresholds
        distances = np.unique(distance_matrix)
        distances.sort()
        
        # If too many distance values, sample them
        if len(distances) > self.filtration_steps:
            indices = np.linspace(0, len(distances)-1, self.filtration_steps, dtype=int)
            distances = distances[indices]
            
        # Track simplices and their birth times
        all_simplices = set()
        simplex_birth = {}
        
        # Track persistence of features
        persistence_pairs = []
        
        # For each dimension, keep track of active features
        active_features = [set() for _ in range(self.homology_dimension_max + 1)]
        
        # Previous Betti numbers
        prev_betti = [0] * (self.homology_dimension_max + 1)
        
        # Compute filtration
        for i, epsilon in enumerate(distances):
            # Compute simplicial complex at this threshold
            new_simplices = self.compute_vietoris_rips_complex(distance_matrix, epsilon)
            
            # Convert to set of tuples for easier manipulation
            new_simplex_set = set(tuple(sorted(simplex)) for simplex in new_simplices)
            
            # Find newly added simplices
            added_simplices = new_simplex_set - all_simplices
            
            # Record birth times for new simplices
            for simplex in added_simplices:
                simplex_birth[simplex] = epsilon
                
            # Update all simplices
            all_simplices = new_simplex_set
            
            # Compute Betti numbers
            betti = self.compute_betti_numbers(new_simplices)
            
            # Record births and deaths based on changes in Betti numbers
            for dim in range(self.homology_dimension_max + 1):
                # Births: Betti number increases
                if betti[dim] > prev_betti[dim]:
                    # Add new active features
                    for _ in range(betti[dim] - prev_betti[dim]):
                        active_features[dim].add((epsilon, None))
                
                # Deaths: Betti number decreases
                elif betti[dim] < prev_betti[dim]:
                    # Remove oldest active features
                    to_remove = []
                    to_add = []
                    
                    # Get oldest features (those born first)
                    features_sorted = sorted(active_features[dim], key=lambda x: x[0])
                    
                    # Mark features that die at this threshold
                    for _ in range(prev_betti[dim] - betti[dim]):
                        if features_sorted:
                            birth, _ = features_sorted.pop(0)
                            to_remove.append((birth, None))
                            to_add.append((birth, epsilon))
                            
                            # Record persistence pair
                            persistence_pairs.append((birth, epsilon, dim))
                    
                    # Update active features
                    for feature in to_remove:
                        active_features[dim].remove(feature)
                        
            # Update previous Betti numbers
            prev_betti = betti
            
        # Features that never die have death time infinity
        for dim in range(self.homology_dimension_max + 1):
            for birth, death in active_features[dim]:
                if death is None:
                    persistence_pairs.append((birth, float('inf'), dim))
                    
        # Sort by persistence (death - birth)
        persistence_pairs.sort(key=lambda x: float('-inf') if x[1] == float('inf') else x[1] - x[0], reverse=True)
        
        # Save result
        self.last_persistence_diagram = persistence_pairs
        
        return persistence_pairs
        
    def identify_significant_features(self, persistence_diagram):
        """
        Identify significant topological features from persistence diagram
        
        Args:
            persistence_diagram: List of (birth, death, dimension) triples
            
        Returns:
            list: List of significant features
        """
        if not persistence_diagram:
            return []
            
        # Calculate persistence of each feature
        features_with_persistence = []
        for birth, death, dim in persistence_diagram:
            if death == float('inf'):
                # Features that never die have maximum persistence
                persistence = float('inf')
            else:
                persistence = death - birth
                
            features_with_persistence.append((birth, death, dim, persistence))
            
        # Sort by persistence
        features_with_persistence.sort(key=lambda x: float('inf') if x[3] == float('inf') else x[3], reverse=True)
        
        # Keep only features with persistence above threshold
        significant_features = []
        for birth, death, dim, persistence in features_with_persistence:
            if persistence > self.persistence_threshold or persistence == float('inf'):
                significant_features.append((birth, death, dim, persistence))
                
        return significant_features
        
    def detect_formation_patterns(self, point_cloud, target_position=None):
        """
        Detect formation patterns from UAV positions
        
        Args:
            point_cloud: List of points (UAV positions)
            target_position: Position of target if available
            
        Returns:
            dict: Detected formation patterns and characteristics
        """
        if len(point_cloud) < 3:
            return {"pattern": "insufficient", "confidence": 0.0}
            
        # Compute persistence diagram
        persistence_diagram = self.compute_persistence_diagram(point_cloud)
        
        # Identify significant features
        significant_features = self.identify_significant_features(persistence_diagram)
        
        # Count features by dimension
        feature_counts = [0] * (self.homology_dimension_max + 1)
        for _, _, dim, _ in significant_features:
            if dim <= self.homology_dimension_max:
                feature_counts[dim] += 1
                
        # Pattern recognition based on topological features
        pattern_data = {}
        
        # β₀ = number of connected components
        # β₁ = number of 1-dimensional holes (loops)
        # β₂ = number of 2-dimensional voids (hollow volumes)
        beta_0 = feature_counts[0]
        beta_1 = feature_counts[1]
        beta_2 = feature_counts[2] if len(feature_counts) > 2 else 0
        
        # Determine formation pattern
        if beta_0 > 1:
            # Multiple disconnected groups
            pattern_data["pattern"] = "fragmented"
            pattern_data["confidence"] = min(1.0, beta_0 / len(point_cloud))
        elif beta_1 >= 1:
            # At least one loop/ring formation
            pattern_data["pattern"] = "encirclement"
            pattern_data["confidence"] = min(1.0, 0.6 + 0.4 * (beta_1 / max(1, len(point_cloud) // 3)))
        elif beta_2 >= 1:
            # 3D enclosure (rare with few UAVs)
            pattern_data["pattern"] = "enclosure"
            pattern_data["confidence"] = min(1.0, 0.7 + 0.3 * beta_2)
        else:
            # No significant topological features
            # Check geometric properties
            
            # Calculate convex hull
            if target_position is not None:
                # Check if target is surrounded
                target_surrounded = self.check_target_surrounded(point_cloud, target_position)
                if target_surrounded:
                    pattern_data["pattern"] = "surrounding"
                    pattern_data["confidence"] = 0.7
                else:
                    # Check if in linear formation
                    linearity = self.calculate_linearity(point_cloud)
                    if linearity > 0.8:
                        pattern_data["pattern"] = "line"
                        pattern_data["confidence"] = linearity
                    else:
                        pattern_data["pattern"] = "dispersed"
                        pattern_data["confidence"] = 0.5
            else:
                # Without target, check general formation
                linearity = self.calculate_linearity(point_cloud)
                if linearity > 0.8:
                    pattern_data["pattern"] = "line"
                    pattern_data["confidence"] = linearity
                else:
                    pattern_data["pattern"] = "cluster"
                    pattern_data["confidence"] = 0.5
                    
        # Update detected patterns
        self.detected_patterns = pattern_data
        
        # Update feature history
        self.feature_history.append(feature_counts)
        if len(self.feature_history) > self.max_history_length:
            self.feature_history.pop(0)
            
        return pattern_data
        
    def check_target_surrounded(self, point_cloud, target_position):
        """
        Check if target is surrounded by UAVs
        
        Args:
            point_cloud: List of points (UAV positions)
            target_position: Position of target
            
        Returns:
            bool: True if target is surrounded
        """
        if len(point_cloud) < 3:
            return False
            
        # Calculate angles from target to each UAV
        angles = []
        for pos in point_cloud:
            dx = pos[0] - target_position[0]
            dy = pos[1] - target_position[1]
            angle = math.atan2(dy, dx)
            angles.append(angle)
            
        # Sort angles
        angles.sort()
        
        # Add first angle again to complete the circle
        angles.append(angles[0] + 2 * math.pi)
        
        # Check maximum gap between consecutive angles
        max_gap = 0
        for i in range(len(angles) - 1):
            gap = angles[i+1] - angles[i]
            max_gap = max(max_gap, gap)
            
        # Target is surrounded if maximum gap is small enough
        # For perfect surrounding, max_gap should be 2π/n where n is number of UAVs
        ideal_gap = 2 * math.pi / len(point_cloud)
        
        # Allow some flexibility - target is surrounded if max gap is not too large
        return max_gap < ideal_gap * 2.5
        
    def calculate_linearity(self, point_cloud):
        """
        Calculate how close points are to forming a line
        
        Args:
            point_cloud: List of points (UAV positions)
            
        Returns:
            float: Linearity measure (0-1), higher is more linear
        """
        if len(point_cloud) < 2:
            return 0.0
            
        # Convert to numpy array
        points = np.array(point_cloud)
        
        # Calculate principal components
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # Calculate covariance matrix
        cov_matrix = np.cov(centered_points.T)
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(cov_matrix)
        
        # Sort eigenvalues
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # For perfect line, only one eigenvalue should be non-zero
        if np.sum(eigenvalues) < 1e-10:
            return 0.0
            
        # Calculate linearity as dominance of first eigenvalue
        linearity = eigenvalues[0] / np.sum(eigenvalues)
        
        return linearity
        
    def calculate_topology_force(self, uav, nearby_uavs, target_position=None):
        """
        Calculate force based on topological analysis of formation
        
        Args:
            uav: The UAV to calculate force for
            nearby_uavs: List of nearby UAVs
            target_position: Position of target if available
            
        Returns:
            numpy.ndarray: The topology force vector [x, y]
        """
        if not nearby_uavs:
            return np.zeros(2)
            
        # Extract positions
        positions = [u.position for u in nearby_uavs + [uav]]
        
        # Detect formation patterns
        pattern_data = self.detect_formation_patterns(positions, target_position)
        pattern = pattern_data["pattern"]
        confidence = pattern_data["confidence"]
        
        # Initialize force
        topology_force = np.zeros(2)
        
        # Get UAV index in position list
        uav_idx = len(nearby_uavs)  # Last index (we appended the UAV's position last)
        
        if pattern == "encirclement" and target_position is not None:
            # Target encirclement - maintain circular formation around target
            
            # Calculate ideal position on circle
            n_uavs = len(positions)
            
            # Sort UAVs by angle around target
            angles = []
            for i, pos in enumerate(positions):
                dx = pos[0] - target_position[0]
                dy = pos[1] - target_position[1]
                angle = math.atan2(dy, dx)
                angles.append((i, angle))
                
            # Sort by angle
            angles.sort(key=lambda x: x[1])
            
            # Find position of this UAV in sorted list
            uav_order = next(i for i, (idx, _) in enumerate(angles) if idx == uav_idx)
            
            # Calculate ideal angle
            ideal_angle = 2 * math.pi * uav_order / n_uavs
            
            # Ideal position on circle
            radius = CONFIG["capture_distance"] / 1.5
            ideal_x = target_position[0] + radius * math.cos(ideal_angle)
            ideal_y = target_position[1] + radius * math.sin(ideal_angle)
            ideal_pos = np.array([ideal_x, ideal_y])
            
            # Force toward ideal position
            direction = ideal_pos - uav.position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                topology_force = direction / distance * CONFIG["uav_max_acceleration"] * confidence
                
        elif pattern == "line":
            # Line formation
            
            # Find line of best fit
            points = np.array(positions)
            centroid = np.mean(points, axis=0)
            centered_points = points - centroid
            
            # Calculate covariance matrix
            cov_matrix = np.cov(centered_points.T)
            
            # Calculate eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Principal direction is eigenvector with largest eigenvalue
            principal_direction = eigenvectors[:, np.argmax(eigenvalues)]
            
            # Project UAV position onto principal direction
            projection = np.dot(uav.position - centroid, principal_direction) * principal_direction
            projection_point = centroid + projection
            
            # Find where this UAV should be in the line
            positions_1d = [np.dot(pos - centroid, principal_direction) for pos in positions]
            positions_1d.sort()
            
            # Get UAV's projected position
            uav_projection = np.dot(uav.position - centroid, principal_direction)
            
            # Find desired position in sorted line
            n_uavs = len(positions)
            uav_rank = min(n_uavs - 1, max(0, int(uav_idx * n_uavs / (n_uavs - 1))))
            desired_projection = positions_1d[uav_rank]
            
            # Desired position on line
            desired_pos = centroid + desired_projection * principal_direction
            
            # Force toward desired position
            direction = desired_pos - uav.position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                topology_force = direction / distance * CONFIG["uav_max_acceleration"] * confidence
                
        elif pattern == "surrounding" and target_position is not None:
            # Target surrounding - similar to encirclement but less structured
            
            # Calculate vector from target to UAV
            direction = uav.position - target_position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                # Normalize
                direction = direction / distance
                
                # Ideal distance
                ideal_distance = CONFIG["capture_distance"] / 1.5
                
                # Force to maintain ideal distance
                radial_force = (ideal_distance - distance) * direction
                
                # Add tangential component for circulation
                tangential_direction = np.array([-direction[1], direction[0]])
                tangential_force = tangential_direction * 0.5
                
                # Combined force
                topology_force = (radial_force + tangential_force) * CONFIG["uav_max_acceleration"] * confidence
                
        elif pattern == "fragmented":
            # Fragmented formations - try to rejoin main group
            
            # Find largest cluster
            clusters = self.find_clusters(positions)
            
            if clusters:
                # Find which cluster this UAV belongs to
                uav_cluster = None
                for cluster in clusters:
                    if uav_idx in cluster:
                        uav_cluster = cluster
                        break
                        
                # If not in largest cluster, move toward largest cluster
                largest_cluster = max(clusters, key=len)
                if uav_cluster is not largest_cluster:
                    # Calculate centroid of largest cluster
                    cluster_positions = [positions[i] for i in largest_cluster]
                    cluster_centroid = np.mean(cluster_positions, axis=0)
                    
                    # Force toward cluster centroid
                    direction = cluster_centroid - uav.position
                    distance = np.linalg.norm(direction)
                    
                    if distance > 0:
                        topology_force = direction / distance * CONFIG["uav_max_acceleration"] * confidence
                        
        else:
            # Default behavior - maintain balanced distribution
            # Calculate centroid
            centroid = np.mean(positions, axis=0)
            
            # Get vector from centroid to UAV
            direction = uav.position - centroid
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                # Normalize
                direction = direction / distance
                
                # Force to maintain moderate distance from centroid
                ideal_distance = CONFIG["uav_radius"] * 5
                topology_force = (ideal_distance - distance) * direction * 0.5 * CONFIG["uav_max_acceleration"]
                
        # Limit force to max acceleration
        force_magnitude = np.linalg.norm(topology_force)
        if force_magnitude > CONFIG["uav_max_acceleration"]:
            topology_force = topology_force / force_magnitude * CONFIG["uav_max_acceleration"]
            
        return topology_force
        
    def find_clusters(self, positions):
        """
        Find clusters of UAVs
        
        Args:
            positions: List of UAV positions
            
        Returns:
            list: List of clusters, each a list of indices
        """
        if not positions:
            return []
            
        # Compute distance matrix
        n_points = len(positions)
        distance_matrix = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.linalg.norm(positions[i] - positions[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                
        # Create graph
        G = nx.Graph()
        
        # Add all vertices
        for i in range(n_points):
            G.add_node(i)
            
        # Add edges where distance is below threshold
        threshold = CONFIG["capture_distance"] / 2
        for i in range(n_points):
            for j in range(i+1, n_points):
                if distance_matrix[i, j] <= threshold:
                    G.add_edge(i, j)
                    
        # Find connected components (clusters)
        clusters = list(nx.connected_components(G))
        
        return clusters
