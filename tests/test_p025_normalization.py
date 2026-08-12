import json

from src.mixed_data.parsers import normalize_manifest_events
from src.mixed_data.source_manifest import load_p025_manifest


def test_normalize_manifest_events_sorts_and_shapes(tmp_path):
    scenario = tmp_path / "dataset" / "scenario"
    (scenario / "zeek").mkdir(parents=True)
    (scenario / "host1").mkdir()
    (scenario / "zeek" / "conn.log").write_text(
        json.dumps(
            {
                "ts": 1753049507.4,
                "uid": "C1",
                "id.orig_h": "128.16.11.13",
                "id.orig_p": 52240,
                "id.resp_h": "128.16.11.6",
                "id.resp_p": 80,
                "proto": "tcp",
                "service": "http",
                "duration": 0.1,
                "orig_pkts": 1,
                "resp_pkts": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (scenario / "zeek" / "http.log").write_text(
        json.dumps(
            {
                "ts": 1753049508.4,
                "uid": "C1",
                "id.orig_h": "128.16.11.13",
                "id.orig_p": 52240,
                "id.resp_h": "128.16.11.6",
                "id.resp_p": 80,
                "method": "GET",
                "host": "128.16.11.6",
                "uri": "/tomcatwar.jsp?cmd=cat /etc/passwd",
                "user_agent": "Wget/1.21.2",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (scenario / "host1" / "auth.log").write_text(
        "Jul 20 13:58:06 host1 useradd[2027]: new user: name=dschroeder, UID=1001\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        """
prototype:
  name: test
  dataset_root: "{dataset_root}"
  scenario_path: "scenario"
  window_seconds: 60
  log_year: 2025
selected_sources:
  - id: zeek_conn
    source_type: zeek_conn
    parser: zeek_json
    role: network_connection
    path: zeek/conn.log
  - id: zeek_http
    source_type: zeek_http
    parser: zeek_json
    role: http_activity
    path: zeek/http.log
  - id: auth
    source_type: host_auth
    parser: linux_auth_text
    role: host_authentication
    path: host1/auth.log
""".format(dataset_root=str(tmp_path / "dataset").replace("\\", "/")),
        encoding="utf-8",
    )

    events = normalize_manifest_events(load_p025_manifest(manifest_path))

    assert [event.source_type for event in events] == ["host_auth", "zeek_conn", "zeek_http"]
    assert events[0].event_type == "user_created"
    assert events[0].username == "dschroeder"
    assert "src_ip" in events[0].missing_fields
    assert events[2].features["has_cmd_param"] == 1
    assert events[2].features["has_passwd_keyword"] == 1
    assert events[2].uid == "C1"
