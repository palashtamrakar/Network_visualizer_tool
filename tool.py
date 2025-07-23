import random
import time
import json
import csv
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QTextEdit, QSlider, QListWidget, QListWidgetItem, QMessageBox,
                             QComboBox, QLineEdit, QTabWidget, QTableWidget, QTableWidgetItem,
                             QFileDialog, QProgressBar, QToolTip)
from PyQt5.QtCore import Qt, QTimer, QTime, QPoint
from PyQt5.QtGui import QFont, QColor, QBrush, QPalette
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure

class DisjointSet:
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, v):
        if self.parent[v] != v:
            self.parent[v] = self.find(self.parent[v])  # Path compression
        return self.parent[v]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] < self.rank[root_v]:
                root_u, root_v = root_v, root_u
            self.parent[root_v] = root_u
            if self.rank[root_u] == self.rank[root_v]:
                self.rank[root_u] += 1
            return True
        return False

def kruskal(nodes, edges):
    mst = []
    ds = DisjointSet(nodes)
    edges.sort(key=lambda x: x[2])  # Sort by weight
    edges_added = 0
    n = len(nodes)
    for u, v, weight in edges:
        if ds.union(u, v) and edges_added < n - 1:
            mst.append((u, v, weight))
            edges_added += 1
            yield mst.copy()
    if mst:
        yield mst

class TCPReno:
    def __init__(self, output_func, animate_func, ack_func, graph_edges, graph_pos, max_window=16):
        self.cwnd = 1
        self.ssthresh = 16  # Initial threshold
        self.max_window = max_window
        self.output_func = output_func
        self.animate_func = animate_func
        self.ack_func = ack_func
        self.edges = graph_edges
        self.pos = graph_pos
        self.nodes = list(set(sum([(u, v) for u, v, _ in graph_edges], ())))
        self.log = []
        self.packet_losses = 0
        self.retransmissions = 0
        self.current_segment = 0
        self.sequence_num = 0
        self.duplicate_acks = 0
        self.rtt = 1.0  # Simulated initial RTT
        self.rto = self.rtt * 2  # Initial RTO
        self.timer = QTimer()
        self.timer.timeout.connect(self.handle_timeout)
        self.loss_prob = 0.2  # Default loss probability
        self.ui_update_func = None  # Callback for UI updates
        self.congestion_node = random.choice(self.nodes)  # Random node to simulate congestion

    def set_ui_update_callback(self, callback):
        """Set the callback function to update the UI."""
        self.ui_update_func = callback

    def send_packets(self, total_segments, rtt_counter=0):
        segment = self.current_segment + 1
        self.output_func(f"[TCP Reno] Sending {self.cwnd} packet(s) for segment {segment}, Seq={self.sequence_num}...")
        src, dst = self.nodes[0], self.nodes[-1]
        packets_sent = 0
        acked_packets = 0
        for i in range(int(self.cwnd)):
            packet_num = packets_sent + 1
            self.sequence_num += 1
            # Simulate congestion at the chosen node
            if self.check_congestion(self.sequence_num):
                self.output_func(f"[Congestion] Detected at node {self.congestion_node}, reducing transmission rate.")
                self.ssthresh = max(self.cwnd // 2, 2)
                self.cwnd = self.ssthresh
                self.log.append(["CA", "Congestion", self.ssthresh, f"{self.cwnd}", "CA"])
                if self.ui_update_func:
                    self.ui_update_func()
                return
            self.animate_func(src, dst, packet_num, self.output_func, segment=segment, color='limegreen')
            if self.packet_loss_simulation():
                self.duplicate_acks += 1
                if self.duplicate_acks == 3:
                    self.output_func(f"3 duplicate ACKs for packet {packet_num} of segment {segment}! Fast retransmit and recovery.")
                    self.ssthresh = max(self.cwnd // 2, 2)
                    self.cwnd = self.ssthresh + 3  # Fast recovery
                    self.log.append(["FR", "Triple Duplicate ACK", self.ssthresh, f"{self.cwnd}", "FR"])
                    self.retransmissions += 1
                    self.duplicate_acks = 0
                    self.sequence_num -= 1  # Retransmit the lost packet
                    self.send_packets(total_segments, rtt_counter)
                    if self.ui_update_func:
                        self.ui_update_func()
                    return
                else:
                    self.timer.start(int(self.rto * 1000))  # Start timeout in ms
            else:
                packets_sent += 1
                acked_packets += 1
                self.timer.stop()  # Stop timer on ACK
                if self.ui_update_func:
                    self.ui_update_func()
        self.output_func(f"All {acked_packets} packets of segment {segment} acknowledged.")
        self.ack_func()
        rtt_counter += 1
        if rtt_counter >= self.rtt:  # Update after RTT
            prev_cwnd = self.cwnd
            if self.cwnd < self.ssthresh:
                self.cwnd *= 2  # Slow start
            else:
                self.cwnd += 1  # Congestion avoidance
            self.cwnd = min(self.cwnd, self.max_window)
            self.log.append(["CA" if prev_cwnd >= self.ssthresh else "SS", "ACK arrived", self.ssthresh, f"{prev_cwnd} -> {self.cwnd}", "CA" if self.cwnd >= self.ssthresh else "SS"])
            if self.ui_update_func:
                self.ui_update_func()
            rtt_counter = 0
        if self.current_segment < total_segments - 1:
            self.current_segment += 1
            self.send_packets(total_segments, rtt_counter)

    def handle_timeout(self):
        self.output_func(f"[TCP Reno] Timeout for Seq={self.sequence_num}. Retransmitting...")
        self.ssthresh = max(self.cwnd // 2, 2)
        self.cwnd = 1
        self.log.append(["SS", "Timeout", self.ssthresh, f"{self.cwnd}", "SS"])
        self.retransmissions += 1
        self.rto *= 2  # Exponential backoff
        self.send_packets(self.current_segment + 1)  # Retransmit current segment
        if self.ui_update_func:
            self.ui_update_func()

    def packet_loss_simulation(self):
        if random.random() < self.loss_prob:
            self.packet_losses += 1
            return True
        return False

    def check_congestion(self, sequence_num):
        # Simulate congestion at the chosen node with a probability
        if random.random() < 0.3 and sequence_num % 5 == 0:  # Arbitrary condition for congestion
            return True
        return False

    def get_stats(self):
        return {"losses": self.packet_losses, "retransmissions": self.retransmissions}

class GraphVisualizer:
    def __init__(self, figure_canvas, ax):
        self.canvas = figure_canvas
        self.ax = ax
        self.G = nx.Graph()
        self.pos = None
        self.speed_factor = 1  # Default speed factor

    def update_graph_structure(self, edges):
        self.G.clear()
        for u, v, w in edges:
            self.G.add_edge(u, v, weight=w)
        self.pos = nx.spring_layout(self.G)

    def draw_graph(self, edges, highlight_edges=None, new_edge=None):
        if not self.pos or len(self.G.edges) != len(edges):
            self.update_graph_structure(edges)
        self.ax.clear()
        weights = nx.get_edge_attributes(self.G, 'weight')
        nx.draw(self.G, self.pos, ax=self.ax, with_labels=True, node_color='skyblue', node_size=600, font_size=10, font_weight='bold')
        # Add tooltips for nodes and edges
        for node in self.G.nodes():
            self.ax.annotate('', xy=self.pos[node], xytext=(0, 0),
                            bbox=dict(boxstyle="round", fc="w", alpha=0.0),
                            ha='center', va='center',
                            annotation_clip=True)
            self.ax.set_title(f"Network Graph - Hover for Details", color='white', fontsize=12, pad=10, backgroundcolor='#2c3e50')
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor('#ecf0f1'))
            QToolTip.setPalette(palette)
            QToolTip.setFont(QFont('Arial', 10))
            self.ax.figure.canvas.mpl_connect('motion_notify_event', lambda event: self._on_hover(event, node, weights.get((node, list(self.G.neighbors(node))[0]) if self.G.neighbors(node) else (node, node), 0)))

        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=weights, ax=self.ax, font_size=8)
        if highlight_edges:
            nx.draw_networkx_edges(self.G, self.pos, edgelist=highlight_edges, edge_color='red', width=2, ax=self.ax)
        if new_edge:
            nx.draw_networkx_edges(self.G, self.pos, edgelist=[new_edge], edge_color='yellow', width=3, alpha=0.7, ax=self.ax)
        self.canvas.draw()

    def _on_hover(self, event, node, edge_weight):
        if event.inaxes and hasattr(self, 'pos'):
            for n, pos in self.pos.items():
                if ((pos[0] - event.xdata) ** 2 + (pos[1] - event.ydata) ** 2) ** 0.5 < 0.1:
                    tooltip_text = f"Node: {n}\nConnected Edge Weight: {edge_weight}"
                    QToolTip.showText(event.globalPos(), tooltip_text, self.canvas)
                    break
            for (u, v), pos in self.pos.items():
                if u in self.G.nodes and v in self.G.nodes:
                    x_mid = (self.pos[u][0] + self.pos[v][0]) / 2
                    y_mid = (self.pos[u][1] + self.pos[v][1]) / 2
                    if ((x_mid - event.xdata) ** 2 + (y_mid - event.ydata) ** 2) ** 0.5 < 0.1:
                        weight = self.G[u][v]['weight']
                        tooltip_text = f"Edge: {u}-{v}\nWeight: {weight}"
                        QToolTip.showText(event.globalPos(), tooltip_text, self.canvas)
                        break

    def animate_packet(self, src, dst, packet_num, output_func, segment=None, color='limegreen'):
        try:
            if src not in self.G or dst not in self.G:
                raise ValueError(f"Source {src} or destination {dst} not in graph")
            path = nx.shortest_path(self.G, source=src, target=dst, weight='weight')
            segment_text = f" (Segment {segment})" if segment else ""
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                start = self.pos[u]
                end = self.pos[v]
                for t in range(21):
                    self.ax.clear()
                    nx.draw(self.G, self.pos, ax=self.ax, with_labels=True, node_color='skyblue', node_size=600, font_size=10, font_weight='bold')
                    nx.draw_networkx_edges(self.G, self.pos, ax=self.ax)
                    x = start[0] + (end[0] - start[0]) * t / 20
                    y = start[1] + (end[1] - start[1]) * t / 20
                    self.ax.add_patch(Rectangle((x - 0.015, y - 0.015), 0.03, 0.03, facecolor=color, edgecolor='black'))
                    self.ax.set_title(f"Transmitting Packet {packet_num}{segment_text}", color='white', fontsize=12, pad=10, backgroundcolor='#2c3e50')
                    self.canvas.draw()
                    QApplication.processEvents()
                    time.sleep(0.01 / self.speed_factor)  # Adjusted sleep time based on speed factor
        except nx.NetworkXNoPath:
            output_func(f"[Error] No path exists between {src} and {dst}")
        except ValueError as e:
            output_func(f"[Error] Animation failed: {e}")
        except Exception as e:
            output_func(f"[Error] Unexpected error during animation: {e}")

    def ack_animation(self):
        self.ax.set_title("ACK Received!", color='green', fontsize=12, pad=10, backgroundcolor='#2c3e50')
        self.canvas.draw()

    def set_speed_factor(self, factor):
        """Set the speed factor for animations."""
        self.speed_factor = factor

class NetworkSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Network Simulator v1.0")
        self.showFullScreen()
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #34495e, stop:1 #2c3e50);
                color: #ecf0f1;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #3498db;
                border-radius: 5px;
                background: #34495e;
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: #ecf0f1;
                padding: 10px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #3498db);
                color: #ffffff;
            }
            QPushButton {
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
            }
            QPushButton#btn_convert, QPushButton#btn_tahoe, QPushButton#btn_stop_wait, QPushButton#btn_export_kruskal, QPushButton#export_button, QPushButton#btn_start, QPushButton#btn_reset, QPushButton#visualize_button, QPushButton#load_button, QPushButton#btn_speed_up {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
            }
            QPushButton#btn_convert:hover, QPushButton#btn_tahoe:hover, QPushButton#btn_stop_wait:hover, QPushButton#btn_export_kruskal:hover, QPushButton#export_button:hover, QPushButton#btn_start:hover, QPushButton#btn_reset:hover, QPushButton#visualize_button:hover, QPushButton#load_button:hover, QPushButton#btn_speed_up:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #3498db);
            }
            QLineEdit {
                background: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 5px;
                padding: 5px;
            }
            QTextEdit {
                background: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 5px;
                padding: 5px;
            }
            QTableWidget {
                background: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 5px;
            }
            QHeaderView::section {
                background: #3498db;
                color: #ecf0f1;
            }
            QProgressBar {
                background: #34495e;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2ecc71, stop:1 #27ae60);
            }
            QSlider {
                background: #34495e;
                border: none;
            }
            QComboBox {
                background: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 5px;
                padding: 5px;
            }
            QLabel {
                font-size: 12px;
                padding: 5px;
            }
        """)
        self.packet_data = []
        self.kruskal_data = ["Data1", "Data2", "Data3", "Data4", "Data5"]
        self.tcp = None
        self.datasets = {
            "Dataset 1 (10 nodes)": (['N' + str(i) for i in range(1, 11)], []),
            "Dataset 2 (13 nodes)": (['N' + str(i) for i in range(1, 14)], []),
            "Dataset 3 (12 nodes)": (['N' + str(i) for i in range(1, 13)], []),
            "Dataset 4 (15 nodes)": (['N' + str(i) for i in range(1, 16)], []),
            "Dataset 5 (11 nodes)": (['N' + str(i) for i in range(1, 12)], [])
        }
        for name, (nodes, edges) in self.datasets.items():
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if random.random() < 0.5:
                        weight = random.randint(1, 10)
                        edges.append((nodes[i], nodes[j], weight))
        self.selected_dataset = None
        self.custom_network_input = None
        self.nodes, self.edges = self.datasets[list(self.datasets.keys())[0]]
        self.current_mst = list(kruskal(self.nodes, self.edges))[-1] if self.edges else []
        self.mst_steps = []
        self.step_index = 0
        self.animation_timer = QTimer()
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: 0; }")
        self.init_tabs()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        self.timeout_duration = 1000  # milliseconds
        self.max_retries = 3
        self.loss_prob = 0.2
        self.current_word = ""
        self.segments_data = {}  # Store segment data for visualization
        self.speed_factor = 1  # Global speed factor for animations

    def init_tabs(self):
        self.init_datagram_tab()
        self.init_datagram_visualization_tab()
        self.init_kruskal_tab()
        self.init_simulation_tab()
        self.init_monitor_tab()

    def init_datagram_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        header = QLabel("TCP Datagram Input & Network Configuration")
        header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #3498db; padding: 10px;")
        layout.addWidget(header)

        network_input_layout = QHBoxLayout()
        self.network_input = QLineEdit()
        self.network_input.setPlaceholderText("Enter custom network (e.g., 'N1-N2-5,N2-N3-3')")
        self.network_input.setToolTip("Enter nodes and edges in format 'N1-N2-weight,N2-N3-weight' (e.g., 'N1-N2-5,N2-N3-3')")
        network_input_layout.addWidget(QLabel("Custom Network:"))
        network_input_layout.addWidget(self.network_input)
        layout.addLayout(network_input_layout)

        load_button = QPushButton("Load Network from File")
        load_button.setObjectName("load_button")
        load_button.setToolTip("Load a network configuration from a CSV or JSON file")
        load_button.clicked.connect(self.load_network_file)
        layout.addWidget(load_button)

        preset_layout = QHBoxLayout()
        self.use_preset = QComboBox()
        self.use_preset.addItems(["No Preset"] + list(self.datasets.keys()))
        self.use_preset.setToolTip("Select a predefined network dataset")
        self.use_preset.currentTextChanged.connect(self.update_selected_dataset)
        preset_layout.addWidget(QLabel("Or Select Preset:"))
        preset_layout.addWidget(self.use_preset)
        layout.addLayout(preset_layout)

        self.preset_table = QTableWidget()
        self.preset_table.setRowCount(5)
        self.preset_table.setColumnCount(3)
        self.preset_table.setHorizontalHeaderLabels(["Dataset", "Nodes", "Edges with Weights"])
        for i, (name, (nodes, edges)) in enumerate(self.datasets.items()):
            self.preset_table.setItem(i, 0, QTableWidgetItem(name))
            self.preset_table.setItem(i, 1, QTableWidgetItem(", ".join(nodes)))
            self.preset_table.setItem(i, 2, QTableWidgetItem(", ".join(f"{u}-{v}-{w}" for u, v, w in edges)))
        self.preset_table.resizeColumnsToContents()
        layout.addWidget(self.preset_table)

        input_layout = QHBoxLayout()
        self.word_input = QLineEdit()
        self.word_input.setPlaceholderText("Enter a word to convert to TCP datagram")
        self.word_input.setToolTip("Enter a word to be segmented into TCP datagrams")
        input_layout.addWidget(QLabel("Input Word:"))
        input_layout.addWidget(self.word_input)
        layout.addLayout(input_layout)

        segments_layout = QHBoxLayout()
        self.segments_input = QLineEdit("4")
        self.segments_input.setPlaceholderText("Number of segments")
        self.segments_input.setToolTip("Number of segments to divide the input word into")
        segments_layout.addWidget(QLabel("Number of Segments:"))
        segments_layout.addWidget(self.segments_input)
        layout.addLayout(segments_layout)

        noise_layout = QHBoxLayout()
        self.noise_input = QLineEdit("20")
        self.noise_input.setPlaceholderText("Channel noise % (0-100)")
        self.noise_input.setToolTip("Percentage of packet loss probability (0-100%)")
        self.noise_input.textChanged.connect(self.update_loss_prob)
        noise_layout.addWidget(QLabel("How noisy is the channel?"))
        noise_layout.addWidget(self.noise_input)
        layout.addLayout(noise_layout)

        encoding_layout = QHBoxLayout()
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems(["ASCII (8-bit)", "UTF-8", "Custom (4-bit)"])
        self.encoding_combo.setToolTip("Select the encoding method for the payload")
        encoding_layout.addWidget(QLabel("Encoding:"))
        encoding_layout.addWidget(self.encoding_combo)
        layout.addLayout(encoding_layout)

        self.btn_convert = QPushButton("Convert & Send to Simulation")
        self.btn_convert.setObjectName("btn_convert")
        self.btn_convert.setToolTip("Convert the input word into datagrams and send to simulation")
        self.btn_convert.clicked.connect(self.convert_and_send)
        layout.addWidget(self.btn_convert)

        self.datagram_output = QTextEdit()
        self.datagram_output.setReadOnly(True)
        layout.addWidget(self.datagram_output)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Datagram Input")

    def init_datagram_visualization_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        header = QLabel("Datagram Visualization")
        header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #3498db; padding: 10px;")
        layout.addWidget(header)

        control_layout = QHBoxLayout()
        self.segment_selector = QComboBox()
        self.segment_selector.setToolTip("Select a segment to visualize its datagram")
        control_layout.addWidget(QLabel("Select Segment:"))
        control_layout.addWidget(self.segment_selector)
        self.visualize_button = QPushButton("Visualize")
        self.visualize_button.setObjectName("visualize_button")
        self.visualize_button.setToolTip("Display the selected segment's datagram structure")
        self.visualize_button.clicked.connect(self.update_visualization)
        control_layout.addWidget(self.visualize_button)
        layout.addLayout(control_layout)

        self.visualization_canvas = FigureCanvas(Figure(facecolor='#34495e'))
        self.visualization_ax = self.visualization_canvas.figure.add_subplot(111)
        self.visualization_ax.set_facecolor('#2c3e50')
        layout.addWidget(self.visualization_canvas)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Datagram Visualization")

    def init_kruskal_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        control_layout = QHBoxLayout()
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(list(self.datasets.keys()))
        self.dataset_combo.setToolTip("Select a dataset for Kruskal's MST simulation")
        self.dataset_combo.currentTextChanged.connect(self.update_selected_dataset)
        control_layout.addWidget(QLabel("Select Dataset:"))
        control_layout.addWidget(self.dataset_combo)

        self.btn_start = QPushButton("Start Simulation")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setToolTip("Start the Kruskal's Minimum Spanning Tree simulation")
        self.btn_start.clicked.connect(self.start_mst_simulation)
        control_layout.addWidget(self.btn_start)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.setToolTip("Reset the Kruskal's MST simulation")
        self.btn_reset.clicked.connect(self.reset_mst_simulation)
        self.btn_reset.setEnabled(False)
        control_layout.addWidget(self.btn_reset)
        layout.addLayout(control_layout)

        self.kruskal_graph_widget = FigureCanvas(Figure(facecolor='#34495e'))
        self.kruskal_ax = self.kruskal_graph_widget.figure.add_subplot(111)
        self.kruskal_ax.set_facecolor('#2c3e50')
        self.kruskal_visualizer = GraphVisualizer(self.kruskal_graph_widget, self.kruskal_ax)
        self.kruskal_visualizer.draw_graph(self.current_mst)
        layout.addWidget(self.kruskal_graph_widget)

        self.kruskal_output = QTextEdit()
        self.kruskal_output.setReadOnly(True)
        layout.addWidget(self.kruskal_output)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Kruskal's MST")

    def init_simulation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        header = QLabel("TCP Reno & Stop-and-Wait Simulator")
        header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #3498db; padding: 10px;")
        layout.addWidget(header)

        graph_layout = QHBoxLayout()
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        graph_layout.addWidget(self.output, 2)

        self.graph_widget = FigureCanvas(Figure(facecolor='#34495e'))
        self.ax = self.graph_widget.figure.add_subplot(111)
        self.ax.set_facecolor('#2c3e50')
        self.graph_visualizer = GraphVisualizer(self.graph_widget, self.ax)
        self.graph_visualizer.draw_graph(self.current_mst)
        graph_layout.addWidget(self.graph_widget, 3)
        layout.addLayout(graph_layout)

        controls_layout = QHBoxLayout()
        self.btn_tahoe = QPushButton("Run TCP Reno")
        self.btn_tahoe.setObjectName("btn_tahoe")
        self.btn_tahoe.setToolTip("Run the TCP Reno congestion control simulation")
        self.btn_tahoe.clicked.connect(self.run_tcp_reno)
        controls_layout.addWidget(self.btn_tahoe)

        self.btn_stop_wait = QPushButton("Run Stop and Wait")
        self.btn_stop_wait.setObjectName("btn_stop_wait")
        self.btn_stop_wait.setToolTip("Run the Stop-and-Wait protocol simulation")
        self.btn_stop_wait.clicked.connect(self.run_stop_and_wait)
        controls_layout.addWidget(self.btn_stop_wait)

        self.btn_export_kruskal = QPushButton("Export Kruskal's Diagram")
        self.btn_export_kruskal.setObjectName("btn_export_kruskal")
        self.btn_export_kruskal.setToolTip("Export the current Kruskal's MST as an image")
        self.btn_export_kruskal.clicked.connect(self.export_kruskal_diagram)
        controls_layout.addWidget(self.btn_export_kruskal)

        self.slider_window = QSlider(Qt.Horizontal)
        self.slider_window.setMinimum(4)
        self.slider_window.setMaximum(64)
        self.slider_window.setValue(16)
        self.slider_window.setToolTip("Adjust the maximum congestion window size (4-64)")
        controls_layout.addWidget(self.slider_window)

        self.mss_input = QLineEdit("16")
        self.mss_input.setToolTip("Maximum Segment Size in bytes")
        self.mtu_input = QLineEdit("1500")
        self.mtu_input.setToolTip("Maximum Transmission Unit in bytes")
        self.domain_type = QComboBox()
        self.domain_type.addItems(["Academic", "Commercial", "Govt"])
        self.domain_type.setToolTip("Select the network domain type")
        self.window_status = QLabel("CWND: -, SSTHRESH: -")
        self.window_status.setStyleSheet("color: #3498db; font-weight: bold;")

        self.timeout_slider = QSlider(Qt.Horizontal)
        self.timeout_slider.setMinimum(500)
        self.timeout_slider.setMaximum(5000)
        self.timeout_slider.setValue(1000)
        self.timeout_slider.setToolTip("Adjust the timeout duration in milliseconds (500-5000)")
        self.timeout_slider.valueChanged.connect(self.update_timeout)
        controls_layout.addWidget(QLabel("Timeout (ms):"))
        controls_layout.addWidget(self.timeout_slider)
        controls_layout.addWidget(QLabel("MSS:"))
        controls_layout.addWidget(self.mss_input)
        controls_layout.addWidget(QLabel("MTU:"))
        controls_layout.addWidget(self.mtu_input)
        controls_layout.addWidget(QLabel("Domain:"))
        controls_layout.addWidget(self.domain_type)
        controls_layout.addWidget(self.window_status)

        self.btn_speed_up = QPushButton("Speed Up Simulation")
        self.btn_speed_up.setObjectName("btn_speed_up")
        self.btn_speed_up.setToolTip("Toggle simulation speed (normal or fast)")
        self.btn_speed_up.clicked.connect(self.toggle_speed)
        controls_layout.addWidget(self.btn_speed_up)

        layout.addLayout(controls_layout)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        label_packets = QLabel("Packets for Stop-and-Wait:")
        layout.addWidget(label_packets)

        self.packet_list = QListWidget()
        for item in self.packet_data:
            QListWidgetItem(item, self.packet_list)
        self.packet_list.itemClicked.connect(self.on_packet_clicked)
        layout.addWidget(self.packet_list)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Simulation")

    def init_monitor_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["State", "Event", "ssthresh", "cwnd", "State"])
        layout.addWidget(self.table)

        self.stats_label = QLabel("Packet Statistics")
        layout.addWidget(self.stats_label)
        self.stats_canvas = FigureCanvas(Figure(facecolor='#34495e'))
        self.stats_ax = self.stats_canvas.figure.add_subplot(111)
        self.stats_ax.set_facecolor('#2c3e50')
        layout.addWidget(self.stats_canvas)

        export_button = QPushButton("Export Logs")
        export_button.setObjectName("export_button")
        export_button.setToolTip("Export simulation logs to a CSV or JSON file")
        export_button.clicked.connect(self.export_simulation_logs)
        layout.addWidget(export_button)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Live Monitor")

    def update_selected_dataset(self, dataset_name):
        if dataset_name in self.datasets:
            self.selected_dataset = dataset_name
            self.nodes, self.edges = self.datasets[dataset_name]
            self.current_mst = list(kruskal(self.nodes, self.edges))[-1] if self.edges else []
            if not self.edges:
                self.kruskal_output.append("[Warning] No edges in the selected dataset.")
            self.kruskal_visualizer.draw_graph(self.current_mst)
            self.graph_visualizer.draw_graph(self.current_mst)
        elif self.custom_network_input:
            self.nodes, self.edges = self.parse_custom_network(self.custom_network_input)
            self.current_mst = list(kruskal(self.nodes, self.edges))[-1] if self.edges else []
            self.kruskal_visualizer.draw_graph(self.current_mst)
            self.graph_visualizer.draw_graph(self.current_mst)
        else:
            self.nodes, self.edges = self.datasets[list(self.datasets.keys())[0]]
            self.current_mst = list(kruskal(self.nodes, self.edges))[-1] if self.edges else []
            self.kruskal_visualizer.draw_graph(self.current_mst)
            self.graph_visualizer.draw_graph(self.current_mst)

    def parse_custom_network(self, input_str):
        if not input_str:
            return [], []
        nodes = set()
        edges = []
        try:
            for edge in input_str.split(","):
                parts = edge.strip().split("-")
                if len(parts) != 3:
                    raise ValueError(f"Invalid edge format: {edge}. Expected format: 'N1-N2-5'")
                u, v, w = parts
                try:
                    w = int(w)
                    if w <= 0:
                        raise ValueError(f"Edge weight must be positive: {w}")
                except ValueError:
                    raise ValueError(f"Invalid weight in edge {edge}: {w}")
                nodes.add(u)
                nodes.add(v)
                edges.append((u, v, w))
        except Exception as e:
            self.datagram_output.append(f"[Error] Invalid network input: {e}")
            return [], []
        return list(nodes), edges

    def load_network_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Network File", "", "CSV Files (*.csv);;JSON Files (*.json)")
        if file_name:
            try:
                if file_name.endswith('.csv'):
                    with open(file_name, 'r') as f:
                        reader = csv.reader(f)
                        next(reader)
                        self.nodes, self.edges = set(), []
                        for row in reader:
                            if len(row) != 3:
                                raise ValueError(f"Invalid row in {file_name}: {row}")
                            u, v, w = row[0], row[1], int(row[2])
                            self.nodes.add(u)
                            self.nodes.add(v)
                            self.edges.append((u, v, w))
                        self.nodes = list(self.nodes)
                elif file_name.endswith('.json'):
                    with open(file_name, 'r') as f:
                        data = json.load(f)
                        self.nodes = data['nodes']
                        self.edges = [(e['source'], e['dest'], e['weight']) for e in data['edges']]
                self.current_mst = list(kruskal(self.nodes, self.edges))[-1] if self.edges else []
                self.datagram_output.append(f"[Info] Loaded network from {file_name}")
                self.kruskal_visualizer.draw_graph(self.current_mst)
                self.graph_visualizer.draw_graph(self.current_mst)
            except Exception as e:
                self.datagram_output.append(f"[Error] Failed to load network: {e}")

    def print_output(self, text):
        self.output.append(text)
        self.window_status.setText(f"CWND: {self.tcp.cwnd if self.tcp else '-'}, SSTHRESH: {self.tcp.ssthresh if self.tcp else '-'}")
        if self.tcp and self.tcp.ui_update_func:
            self.tcp.ui_update_func()  # Trigger UI update on each output

    def print_kruskal_output(self, text):
        self.kruskal_output.append(text)

    def compute_mst_stats(self):
        if not self.current_mst:
            return "No MST available"
        total_weight = sum(w for _, _, w in self.current_mst)
        num_edges = len(self.current_mst)
        G = nx.Graph()
        G.add_weighted_edges_from(self.current_mst)
        avg_clustering = nx.average_clustering(G)
        return (
            f"[MST Statistics]\n"
            f"Total Weight: {total_weight}\n"
            f"Number of Edges: {num_edges}\n"
            f"Average Clustering Coefficient: {avg_clustering:.3f}"
        )

    def start_mst_simulation(self):
        self.kruskal_output.clear()
        self.print_kruskal_output("[Kruskal] Starting Minimum Spanning Tree Simulation:")
        self.mst_steps = list(kruskal(self.nodes, self.edges))
        self.step_index = 0
        self.current_mst = []
        self.kruskal_visualizer.draw_graph(self.edges)
        self.btn_start.setEnabled(False)
        self.btn_reset.setEnabled(True)
        self.animation_timer.timeout.connect(self.animate_mst_step)
        self.animation_timer.start(1000)

    def animate_mst_step(self):
        if self.step_index < len(self.mst_steps):
            self.current_mst = self.mst_steps[self.step_index]
            new_edge = self.current_mst[-1][:2]
            self.kruskal_visualizer.draw_graph(self.edges, highlight_edges=[(u, v) for u, v, _ in self.current_mst[:-1]], new_edge=new_edge)
            for u, v, w in self.current_mst[-1:]:
                self.print_kruskal_output(f"Step {self.step_index + 1}: Adding edge {u}-{v} with weight {w}")
            self.step_index += 1
            QApplication.processEvents()
            time.sleep(0.5 / self.speed_factor)
            self.kruskal_visualizer.draw_graph(self.edges, highlight_edges=[(u, v) for u, v, _ in self.current_mst])
        else:
            self.print_kruskal_output("[Kruskal] MST Complete!")
            self.print_kruskal_output(self.compute_mst_stats())
            self.graph_visualizer.draw_graph(self.current_mst)
            self.animation_timer.stop()
            self.btn_reset.setEnabled(True)

    def reset_mst_simulation(self):
        self.step_index = 0
        self.current_mst = []
        self.mst_steps = []  # Clear steps to free memory
        self.kruskal_output.clear()
        self.kruskal_visualizer.draw_graph(self.edges)
        self.btn_start.setEnabled(True)
        self.btn_reset.setEnabled(False)
        self.animation_timer.stop()

    def convert_and_send(self):
        custom_input = self.network_input.text().strip()
        if custom_input:
            self.custom_network_input = custom_input
            self.nodes, self.edges = self.parse_custom_network(custom_input)
            self.current_mst = list(kruskal(self.nodes, self.edges))[-1] if self.edges else []
            self.datagram_output.append(f"[Info] Custom network set: {custom_input}")
        elif self.use_preset.currentText() != "No Preset":
            self.selected_dataset = self.use_preset.currentText()
            self.nodes, self.edges = self.datasets[self.selected_dataset]
            self.current_mst = list(kruskal(self.nodes, self.edges))[-1] if self.edges else []
            self.datagram_output.append(f"[Info] Using preset: {self.selected_dataset}")
        else:
            self.nodes, self.edges = self.datasets[list(self.datasets.keys())[0]]
            self.current_mst = list(kruskal(self.nodes, self.edges))[-1] if self.edges else []
            self.datagram_output.append("[Info] Using default preset: Dataset 1 (10 nodes)")

        word = self.word_input.text().strip()
        if not word:
            self.datagram_output.append("[Error] Please enter a word.")
            return
        self.current_word = word  # Store the original word
        self.segments_data.clear()  # Clear previous segment data
        try:
            mss = int(self.mss_input.text() or "16")
            if mss <= 0:
                raise ValueError("MSS must be positive")
            segments_count = int(self.segments_input.text() or "4")
            if segments_count <= 0:
                raise ValueError("Number of segments must be positive")
            encoding = self.encoding_combo.currentText()
            if encoding == "ASCII (8-bit)":
                payload_bin = ''.join(format(ord(c), '08b') for c in word)
            elif encoding == "UTF-8":
                payload_bin = ''.join(format(c, '08b') for c in word.encode('utf-8'))
            else:
                payload_bin = ''.join(format(ord(c) % 16, '04b') for c in word)
            segment_size = max(len(payload_bin) // segments_count, (mss * 8) // segments_count)
            segments = [payload_bin[i:i + segment_size] for i in range(0, len(payload_bin), segment_size)]
            while len(segments) < segments_count:
                segments.append('')
            char_segments = [word[i // 8:(i + segment_size) // 8] for i in range(0, len(word) * 8, segment_size)]
            while len(char_segments) < segments_count:
                char_segments.append('')
            self.packet_data = [f"{word} (Segment {i+1})" for i in range(segments_count)]
            self.packet_list.clear()
            for item in self.packet_data:
                QListWidgetItem(item, self.packet_list)
            for i, (segment, char_segment) in enumerate(zip(segments, char_segments)):
                header_bin = ''.join(random.choice('01') for _ in range(20))  # Initial 20 bits for compatibility
                self.segments_data[i + 1] = {"header": header_bin, "payload": segment, "text": char_segment}
                datagram = (
                    f"[Datagram View - Segment {i+1}]\n"
                    f"Header (20 bits):   {header_bin}\n"
                    f"Payload ({len(segment)} bits): {segment}\n"
                    f"Text Payload: '{char_segment}'\n"
                    f"Total Length: {20 + len(segment)} bits"
                )
                component_list = (
                    f"\n[Datagram Components]\n"
                    f"- Header: 20-bit binary string\n"
                    f"- Payload: Binary data of input word '{word}' (Segment {i+1}: '{char_segment}')\n"
                    f"- Total Length: Header + Payload bits"
                )
                self.datagram_output.append(datagram + component_list)
            self.segment_selector.clear()
            self.segment_selector.addItems([str(i) for i in range(1, segments_count + 1)])
            self.datagram_output.append(f"[\u2713] Word '{word}' segmented into {len(segments)} packets and sent to Simulation tab.")
        except ValueError as e:
            self.datagram_output.append(f"[Error] Invalid MSS, segments, or encoding: {e}")

    def update_visualization(self):
        if not self.segments_data:
            return
        segment_num = int(self.segment_selector.currentText())
        if segment_num in self.segments_data:
            header = self.segments_data[segment_num]["header"]
            payload = self.segments_data[segment_num]["payload"]
            text = self.segments_data[segment_num]["text"]
            self.visualize_datagram(segment_num, header, payload, self.current_word, text)

    def visualize_datagram(self, segment_num, header, payload, full_word, segment_text):
        self.visualization_ax.clear()
        self.visualization_ax.set_facecolor('#2c3e50')
        self.visualization_ax.set_title(f"Datagram for '{full_word}' - Segment {segment_num}", color='#ecf0f1', fontsize=14, pad=10)

        # Simulate IP Header (20 bytes = 160 bits)
        ip_header = ''.join(random.choice('01') for _ in range(160))
        ip_ver_len = ip_header[:8]  # 4 bits version, 4 bits length
        src_ip = ip_header[8:40]  # 32 bits source IP
        dst_ip = ip_header[40:72]  # 32 bits destination IP
        ip_text = (
            f"IP Header (160 bits)\n"
            f"Version/Length: {ip_ver_len}\n"
            f"Src IP: {src_ip[:8]}...{src_ip[-8:]}\n"
            f"Dst IP: {dst_ip[:8]}...{dst_ip[-8:]}"
        )
        ip_rect = Rectangle((0.1, 0.8), 0.8, 0.1, facecolor='#8e44ad', edgecolor='#ecf0f1', linewidth=1.5)
        self.visualization_ax.add_patch(ip_rect)
        self.visualization_ax.text(0.5, 0.85, "Head", ha='center', va='center', color='#ecf0f1', fontsize=6)
        self.visualization_ax.figure.canvas.mpl_connect('motion_notify_event', lambda event: self._show_tooltip(event, ip_rect, ip_text))

        # Draw TCP header
        tcp_header = header + ''.join(random.choice('01') for _ in range(140))  # Extend to 160 bits
        src_port = tcp_header[:16]  # 16 bits
        dst_port = tcp_header[16:32]  # 16 bits
        seq_num = tcp_header[32:64]  # 32 bits
        ack_num = tcp_header[64:96]  # 32 bits
        data_offset = tcp_header[96:100]  # 4 bits
        flags = tcp_header[100:106]  # 6 bits
        window = tcp_header[106:122]  # 16 bits
        checksum = tcp_header[122:138]  # 16 bits
        urgent = tcp_header[138:160]  # 22 bits (padded to 160)
        header_text = (
            f"TCP Header (160 bits)\n"
            f"Src Port: {src_port}\n"
            f"Dst Port: {dst_port}\n"
            f"Seq Num: {seq_num[:8]}...{seq_num[-8:]}\n"
            f"Ack Num: {ack_num[:8]}...{ack_num[-8:]}\n"
            f"Data Offset: {data_offset}\n"
            f"Flags: {flags} (URG={flags[0]}, ACK={flags[1]}, PSH={flags[2]}, RST={flags[3]}, SYN={flags[4]}, FIN={flags[5]})\n"
            f"Window: {window}\n"
            f"Checksum: {checksum}\n"
            f"Urgent Ptr: {urgent[:8]}...{urgent[-8:]}"
        )
        header_rect = Rectangle((0.1, 0.6), 0.8, 0.15, facecolor='#3498db', edgecolor='#ecf0f1', linewidth=1.5)
        self.visualization_ax.add_patch(header_rect)
        self.visualization_ax.text(0.5, 0.675, "Tailer", ha='center', va='center', color='#ecf0f1', fontsize=6)
        self.visualization_ax.figure.canvas.mpl_connect('motion_notify_event', lambda event: self._show_tooltip(event, header_rect, header_text))

        # Draw payload
        payload_text = ""
        encoding = self.encoding_combo.currentText()
        if encoding == "ASCII (8-bit)":
            for i in range(0, len(payload), 8):
                byte = payload[i:i+8]
                if len(byte) == 8:
                    payload_text += chr(int(byte, 2))
        elif encoding == "UTF-8":
            i = 0
            while i < len(payload):
                byte_count = 1
                first_byte = payload[i:i+8]
                if len(first_byte) == 8:
                    if first_byte.startswith('0'):  # Single-byte character
                        payload_text += chr(int(first_byte, 2))
                    elif first_byte.startswith('110') and i + 16 <= len(payload):  # Two-byte character
                        byte_count = 2
                        bytes_str = payload[i:i+16]
                        try:
                            payload_text += bytes([int(bytes_str[:8], 2), int(bytes_str[8:16], 2)]).decode('utf-8', errors='ignore')
                        except:
                            payload_text += '?'
                    else:
                        payload_text += '?'
                    i += byte_count * 8
                else:
                    break
        else:  # Custom (4-bit)
            for i in range(0, len(payload), 4):
                nibble = payload[i:i+4]
                if len(nibble) == 4:
                    payload_text += chr(int(nibble, 2) + 32)  # Map 0-15 to ASCII printable
        payload_display = (
            f"Payload ({len(payload)} bits)\n"
            f"Binary: {payload}\n"
            f"Text: '{payload_text or segment_text}'"
        )
        payload_rect = Rectangle((0.1, 0.3), 0.8, 0.2, facecolor='#2ecc71', edgecolor='#ecf0f1', linewidth=1.5)
        self.visualization_ax.add_patch(payload_rect)
        self.visualization_ax.text(0.5, 0.4, "Data", ha='center', va='center', color='#ecf0f1', fontsize=6)
        self.visualization_ax.figure.canvas.mpl_connect('motion_notify_event', lambda event: self._show_tooltip(event, payload_rect, payload_display))

        self.visualization_ax.text(0.5, 0.1, f"Full Word: '{full_word}'\nTotal Length: {160 + 160 + len(payload)} bits", ha='center', va='center', color='#ecf0f1', fontsize=10, bbox=dict(facecolor='#34495e', edgecolor='#ecf0f1', boxstyle='round'))

        self.visualization_ax.set_xlim(0, 1)
        self.visualization_ax.set_ylim(0, 1)
        self.visualization_ax.axis('off')
        self.visualization_canvas.draw()

    def _show_tooltip(self, event, rect, text):
        if event.inaxes and rect.contains(event)[0]:
            canvas_pos = self.visualization_canvas.mapToGlobal(QPoint(int(event.x), int(event.y)))
            QToolTip.showText(canvas_pos, text, self.visualization_canvas)

    def run_tcp_reno(self):
        self.output.clear()
        if not self.current_mst or not self.graph_visualizer:
            self.print_output("[Error] No valid network graph or visualizer available. Please load or select a dataset.")
            return
        try:
            max_window = self.slider_window.value()
            mss = int(self.mss_input.text() or "16")
            mtu = int(self.mtu_input.text() or "1500")
            if mss <= 0 or mtu <= 0:
                raise ValueError("MSS and MTU must be positive integers")
            self.tcp = TCPReno(self.print_output, self.graph_visualizer.animate_packet, self.graph_visualizer.ack_animation, self.current_mst, self.graph_visualizer.pos, max_window)
            self.tcp.loss_prob = self.loss_prob
            self.tcp.set_ui_update_callback(self.update_monitor)  # Set the callback
            total_segments = len(self.packet_data)
            self.progress_bar.setRange(0, total_segments * 8)
            self.graph_visualizer.set_speed_factor(self.speed_factor)  # Set initial speed
            self.tcp.send_packets(total_segments)
            self.progress_bar.setValue(total_segments * 8)
            self.update_monitor()  # Final update
            self.print_output("[TCP Reno] Simulation completed for all segments.")
        except ValueError as e:
            self.print_output(f"[Error] Invalid input: {e}")
        except Exception as e:
            self.print_output(f"[Error] Unexpected error: {e}")

    def run_stop_and_wait(self):
        self.output.clear()
        if not self.packet_list.count() or not self.graph_visualizer:
            self.print_output("[Error] No packets available or visualizer not initialized. Please convert a word first.")
            return
        packets = [self.packet_list.item(i).text() for i in range(self.packet_list.count())]
        src, dst = self.nodes[0], self.nodes[-1]
        self.progress_bar.setRange(0, len(packets))
        self.timer = QTimer(self)
        self.rtt = 1.0  # Simulated initial RTT
        self.timeout_duration = self.rtt * 2 * 1000  # Initial timeout in ms
        self.current_packet = 0
        self.sequence = 0
        self.attempts = 0
        self.timer.timeout.connect(lambda: self.handle_timeout(packets, src, dst))
        self.graph_visualizer.set_speed_factor(self.speed_factor)  # Set initial speed
        self.send_packet(packets, src, dst)

    def send_packet(self, packets, src, dst):
        if self.current_packet >= len(packets):
            self.timer.stop()
            self.print_output("[Stop-and-Wait] Simulation completed.")
            return
        pkt = packets[self.current_packet]
        segment_num = int(pkt.split("Segment ")[1].split(")")[0])
        self.attempts += 1
        self.print_output(f"[Stop-and-Wait] Sending packet for {pkt} (Attempt {self.attempts}, Seq={self.sequence})")
        self.graph_visualizer.animate_packet(src, dst, self.current_packet + 1, self.print_output, segment=segment_num)
        if random.random() > self.loss_prob:
            self.print_output(f"[Stop-and-Wait] ACK received for Seq={self.sequence}")
            self.graph_visualizer.ack_animation()
            expected_seq = self.sequence  # ACK matches sent sequence
            if expected_seq == self.sequence:
                self.sequence = 1 - self.sequence  # Toggle sequence number
                self.current_packet += 1
                self.attempts = 0
                self.rtt = self.rtt * 0.875 + 0.125 * 1.0  # Simple RTT update
                self.timeout_duration = self.rtt * 2 * 1000
                self.progress_bar.setValue(self.current_packet)
                self.timer.stop()
                self.send_packet(packets, src, dst)
        else:
            self.timer.start(self.timeout_duration)

    def handle_timeout(self, packets, src, dst):
        if self.attempts < self.max_retries:
            self.timeout_duration *= 2  # Exponential backoff
            self.print_output(f"[Stop-and-Wait] Timeout. Resending {packets[self.current_packet]} (Seq={self.sequence})")
            self.send_packet(packets, src, dst)
        else:
            self.print_output(f"[Stop-and-Wait] Failed to deliver {packets[self.current_packet]} after {self.max_retries} attempts.")
            self.current_packet += 1
            self.attempts = 0
            self.timeout_duration = self.rtt * 2 * 1000  # Reset timeout
            self.progress_bar.setValue(self.current_packet)
            self.timer.stop()
            self.send_packet(packets, src, dst)

    def export_kruskal_diagram(self):
        if not self.current_mst or not self.graph_visualizer:
            self.print_output("[Error] No MST available to export. Please run a simulation first.")
            return
        try:
            self.graph_widget.figure.savefig("kruskal_mst.png", dpi=300, bbox_inches='tight')
            self.print_output("[\u2713] Kruskal's MST exported as 'kruskal_mst.png'")
        except Exception as e:
            self.print_output(f"[Error] Could not export Kruskal's diagram: {e}")

    def export_simulation_logs(self):
        if not self.tcp or not self.tcp.log:
            self.print_output("[Error] No simulation logs available.")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Logs", "", "CSV Files (*.csv);;JSON Files (*.json)")
        if file_name:
            try:
                if file_name.endswith('.csv'):
                    with open(file_name, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["State", "Event", "ssthresh", "cwnd", "State"])
                        writer.writerows(self.tcp.log)
                elif file_name.endswith('.json'):
                    with open(file_name, 'w') as f:
                        json.dump({"logs": self.tcp.log, "mst": self.current_mst}, f, indent=2)
                self.print_output(f"[\u2713] Logs exported to {file_name}")
            except Exception as e:
                self.print_output(f"[Error] Failed to export logs: {e}")

    def update_monitor(self):
        """Update the Live Monitor table and stats in real-time."""
        if self.tcp is None or not self.tcp.log:
            return
        self.table.setRowCount(len(self.tcp.log))
        for i, row in enumerate(self.tcp.log):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setForeground(QBrush(QColor("#ecf0f1")))
                item.setBackground(QBrush(QColor("#34495e")))
                self.table.setItem(i, j, item)
        stats = self.tcp.get_stats()
        self.stats_ax.clear()
        self.stats_ax.set_facecolor('#2c3e50')
        labels = ["Packet Losses", "Retransmissions"]
        values = [stats["losses"], stats["retransmissions"]]
        self.stats_ax.bar(labels, values, color=['#FF6384', '#36A2EB'], edgecolor='#ecf0f1')
        self.stats_ax.set_ylabel("Count", color='#ecf0f1')
        self.stats_ax.set_title("TCP Statistics", color='#ecf0f1')
        self.stats_ax.tick_params(colors='#ecf0f1')
        self.stats_canvas.draw()

    def on_packet_clicked(self, item):
        packet_data = item.text()
        self.print_output(f"[App Layer] Data received from network layer: {packet_data}")
        encoding = self.encoding_combo.currentText()
        if encoding == "ASCII (8-bit)":
            payload_bin = ''.join(format(ord(c), '08b') for c in packet_data.split(' (')[0])
        elif encoding == "UTF-8":
            payload_bin = ''.join(format(c, '08b') for c in packet_data.split(' (')[0].encode('utf-8'))
        else:
            payload_bin = ''.join(format(ord(c) % 16, '04b') for c in packet_data.split(' (')[0])
        header_bin = ''.join(random.choice('01') for _ in range(20))
        datagram = (
            f"[Datagram View]\n"
            f"Header (20 bits):   {header_bin}\n"
            f"Payload ({len(payload_bin)} bits): {payload_bin}\n"
            f"Total Length: {20 + len(payload_bin)} bits"
        )
        component_list = (
            f"\n[Datagram Components]\n"
            f"- Header: 20-bit binary string\n"
            f"- Payload: Binary data of input word '{packet_data}'\n"
            f"- Total Length: Header + Payload bits"
        )
        self.print_output(datagram + component_list)

    def on_kruskal_packet_clicked(self, item):
        packet_data = item.text()
        self.print_kruskal_output(f"[App Layer] Data received from network layer: {packet_data}")
        encoding = self.encoding_combo.currentText()
        if encoding == "ASCII (8-bit)":
            payload_bin = ''.join(format(ord(c), '08b') for c in packet_data)
        elif encoding == "UTF-8":
            payload_bin = ''.join(format(c, '08b') for c in packet_data.encode('utf-8'))
        else:
            payload_bin = ''.join(format(ord(c) % 16, '04b') for c in packet_data)
        header_bin = ''.join(random.choice('01') for _ in range(20))
        datagram = (
            f"[Datagram View]\n"
            f"Header (20 bits):   {header_bin}\n"
            f"Payload ({len(payload_bin)} bits): {payload_bin}\n"
            f"Total Length: {20 + len(payload_bin)} bits"
        )
        component_list = (
            f"\n[Datagram Components]\n"
            f"- Header: 20-bit binary string\n"
            f"- Payload: Binary data of input word '{packet_data}'\n"
            f"- Total Length: Header + Payload bits"
        )
        self.print_kruskal_output(datagram + component_list)

    def update_loss_prob(self):
        try:
            noise = float(self.noise_input.text() or "20")
            if 0 <= noise <= 100:
                self.loss_prob = noise / 100.0
                if self.tcp:
                    self.tcp.loss_prob = self.loss_prob
            else:
                self.datagram_output.append("[Error] Noise percentage must be between 0 and 100.")
        except ValueError:
            self.datagram_output.append("[Error] Please enter a valid number for channel noise.")

    def update_timeout(self, value):
        self.timeout_duration = value

    def toggle_speed(self):
        """Toggle between normal and fast simulation speed."""
        if self.speed_factor == 1:
            self.speed_factor = 5  # Faster speed
            self.graph_visualizer.set_speed_factor(5)
            self.btn_speed_up.setText("Slow Down Simulation")
            self.print_output("[Info] Simulation speed increased (x5).")
        else:
            self.speed_factor = 1  # Normal speed
            self.graph_visualizer.set_speed_factor(1)
            self.btn_speed_up.setText("Speed Up Simulation")
            self.print_output("[Info] Simulation speed reset to normal.")
        # Update all visualizers if needed
        if self.kruskal_visualizer:
            self.kruskal_visualizer.set_speed_factor(self.speed_factor)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = NetworkSimulator()
    window.show()
    sys.exit(app.exec_())
