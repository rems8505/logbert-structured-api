import gradio as gr
import requests
import json
import time
import tempfile

WINDOW = 32
API = "http://localhost:8000/score_batch"


def process_log(file_obj, sleep_time):
    # Read uploaded or example file
    if hasattr(file_obj, "read"):
        lines = file_obj.read().decode("utf-8").splitlines()
    else:
        with open(file_obj, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    output_lines = []
    output_file_path = "output_anomalies.txt"

    with open(output_file_path, "w", encoding="utf-8") as fout:
        for i in range(0, len(lines), 32):
            batch = lines[i:i+32]
            if len(batch) < 32:
                break

            try:
                response = requests.post(
                    "http://localhost:8000/score_batch",
                    data=json.dumps({"lines": batch})
                )
                response.raise_for_status()
                ans = response.json()

                if ans.get("is_anomaly"):
                    hdr_parts = [f"reason={ans['reason']}",
                                 f"score={ans.get('score', 1):.4f}"]

                    if ans["reason"] == "unseen_template":
                        hdr_parts.append(f"cluster={ans['cluster_id']}")
                    else:  # miss_ratio
                        hdr_parts.append(
                            f"miss={ans['miss_count']}/{ans['masked']}  "
                            f"g={ans['g']} r={ans['r']}"
                        )

                    header = " ".join(hdr_parts)
                    start_line = i + 1
                    end_line = i + len(batch)

                    output_lines.append(f"🚨 [ALERT] lines {start_line}-{end_line}  {header}")
                    fout.write(f"# ---- anomaly lines {start_line}-{end_line}  {header} ----\n")
                    fout.write("\n".join(batch) + "\n\n")
                else:
                    output_lines.append(f"✅ lines {i+1}-{i+len(batch)} OK")

            except Exception as e:
                output_lines.append(f"❌ Error in lines {i+1}-{i+len(batch)}: {str(e)}")

            time.sleep(sleep_time)

    return "\n".join(output_lines), output_file_path


# Sample input file for example
example_file = "openstack_structured.csv"  # <- Make sure this file exists in the same folder

demo = gr.Interface(
    fn=process_log,
    inputs=[
        gr.File(label="Upload Log File (.csv)", file_types=[".csv"]),
        gr.Slider(0, 3, value=1, step=0.1, label="Sleep Time Between Batches (seconds)")
    ],
    outputs=[
        gr.Textbox(label="Batch-wise Results"),
        gr.File(label="Anomaly Output File")
    ],
    examples=[[example_file, 0.5]],
    title="Log Anomaly Detector (Gradio UI)",
    description="This UI takes a log file, sends logs in batches to the inference API (`/score_batch`), and marks anomalies."
)

if __name__ == "__main__":
    demo.launch()



gr.Interface(
    fn=process_log,
    inputs=[
        gr.File(label="Upload Log File"),
        gr.Slider(0, 2, value=0, step=0.1, label="Sleep between batches (sec)")
    ],
    outputs=[
        gr.Textbox(label="Anomaly Detection Output", lines=20),
        gr.File(label="Download Anomalous Batches")
    ],
    title="LogBERT Anomaly Detector (Batch of 32)",
    description="Uploads a raw log file and checks for anomalies in 32-line batches using the LogBERT inference server."
).launch()
