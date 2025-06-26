import os
import xml.etree.ElementTree as ET
import networkx as nx

def parse_dax_xml_to_gml(xml_file_path, output_dir):
    G = nx.DiGraph()
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    # Namespace for Pegasus DAX XML
    ns = {"dax": "http://pegasus.isi.edu/schema/DAX"}
    # Map output files to their producing job
    file_producer = {}
    for job in root.findall("dax:job", ns):
        job_id = job.attrib["id"]
        runtime = float(job.attrib.get("runtime", 10.0))
        G.add_node(job_id, LengthOnScheduledMachine=runtime)

        for use in job.findall("dax:uses", ns):
            file = use.attrib["file"]
            link_type = use.attrib["link"]
            if link_type == "output":
                file_producer[file] = job_id
            # Add edges for input dependencies
    for job in root.findall("dax:job", ns):
        job_id = job.attrib["id"]
        for use in job.findall("dax:uses", ns):
            file = use.attrib["file"]
            if use.attrib["link"] == "input" and file in file_producer:
                producer_job = file_producer[file]
                if producer_job != job_id:
                    G.add_edge(producer_job, job_id)
    # Write to GML
    base_name = os.path.splitext(os.path.basename(xml_file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.gml")
    os.makedirs(output_dir, exist_ok=True)
    nx.write_gml(G, output_path)
    print(f"✅ Saved {output_path}")

    
if __name__ == "__main__":
    input_folder = "C:/Users/palas/OneDrive/Desktop/DAD_Computing/TaskOffloadingOptimization/data/SIPHT" # Update path
    output_folder = "C:/Users/palas/OneDrive/Desktop/DAD_Computing/TaskOffloadingOptimization/data/SIPHT_dags"
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.endswith(".xml"):
            xml_path = os.path.join(input_folder, file)
            parse_dax_xml_to_gml(xml_path, output_folder)
