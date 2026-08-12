# Addressing Challenges

These notes summarize how the mixed-data Watcher Agent prototype addresses the main engineering challenges, and what still needs future work.

## Points Covered

- Missing data should be explicit and encoded as signal.
- Source meaning should be preserved with `source_type` and source roles.
- Feature encoding must convert mixed raw fields into stable numeric inputs for Mamba.
- Raw readable evidence should still be preserved for the LLM.
- Stable identifiers should remain recognizable outside the Mamba representation.
- Mamba should model behavior over time, not replace original facts like IPs or usernames.
- Fixed 60-second windows are acceptable for the prototype, but the windowing strategy needs later review.
- Event relationships are useful, but not mandatory for the first prototype.
- App-level stream memory is sufficient for now, but it is not true recurrent Mamba hidden-state carryover.
- The current Mamba encoder path proves the plumbing, not security intelligence.
- Mamba still needs a meaningful learning objective later, even if it is used only as an encoder.
- Duplicate evidence, timestamp alignment, event granularity, volume, labeling, formal evaluation, trust, and context selection are currently scoped as prototype assumptions.

## Missing Data

- Missing data should not be silently hidden or converted into fake values.
- Nullable fields such as `src_ip`, `dst_ip`, `username`, `uri`, `hash`, or `signature` should remain visibly nullable in the readable packet.
- Each event should include `missing_fields` and `available_fields`.
- For Mamba, missingness must become numeric signal through flags such as `src_ip_present`, `username_present`, `uri_present`, or `hash_present`.
- This prevents the model from confusing "missing" with a real numeric value such as `0`.
- Missingness can be useful evidence because some sources naturally lack certain fields, while unusual missingness may indicate weak visibility or incomplete telemetry.

## Source Meaning

- Every normalized event should keep a `source_type`.
- Current prototype source types include `zeek_conn`, `zeek_http`, and `host_auth`.
- Future source types may include `zeek_dns`, `zeek_ssh`, `zeek_files`, `suricata_alert`, `host_syslog`, `endpoint_telemetry`, and `pcap_packet`.
- Adding `source_type` does not create more rows. It adds meaning to each existing row.
- Combining more files creates more events, but `source_type` helps the system understand where each event came from.
- For the LLM, `source_type` can remain readable text.
- For Mamba, `source_type` should be encoded numerically through one-hot encoding or learned embeddings.
- A deeper taxonomy such as `sensor_type`, `evidence_level`, or `network_layer` can be added later if needed.

## Feature Encoding

- The shared event shape can contain numeric values, strings, IP addresses, ports, URLs, usernames, hashes, commands, and free-text logs.
- Mamba cannot directly reason over raw strings, so the Mamba input must be a consistent numeric vector.
- Numeric fields can be normalized.
- Categorical fields such as `source_type`, protocol, service, and event type can be one-hot encoded or embedded.
- Text fields such as URI, command, user agent, and log message can start with simple numeric features.
- Useful simple features include length, keyword flags, suspicious-token counts, file-extension flags, and presence flags.
- Advanced text embeddings are possible later, but they are not required for the first prototype.

## P025 Encoding Issues

- Zeek connection logs contain IP fields such as `id.orig_h` and `id.resp_h`.
- IP addresses should not be converted into one large raw number.
- Better prototype features include `src_ip_internal`, `dst_ip_internal`, `same_subnet`, `src_host_id`, `dst_host_id`, and common port flags.
- Zeek HTTP logs contain URI strings such as `/tomcatwar.jsp?pwd=j&cmd=cat /etc/passwd`.
- Better URI features include `uri_length`, `has_query_string`, `has_cmd_param`, `has_passwd_keyword`, `has_shadow_keyword`, `path_depth`, and file-extension flags such as `file_extension_jsp`.
- HTTP user agents such as `Wget/1.21.2` or `python-requests/2.32.3` should become flags such as `user_agent_wget`, `user_agent_python_requests`, `user_agent_browser_like`, `user_agent_cli_tool`, and `user_agent_missing`.
- File hashes such as `md5`, `sha1`, and `sha256` should not be treated as raw numbers.
- Better hash/file features include `file_hash_present`, `seen_bytes`, `total_bytes`, MIME type flags, `file_seen_before`, and `same_hash_repeated`.
- Host auth logs contain free-text messages about user creation, password changes, sudo sessions, cron sessions, and commands.
- Better host-log features include `event_user_created`, `event_password_changed`, `event_sudo_used`, `event_ssh_keygen`, `event_session_opened`, `event_session_closed`, `username_present`, and `command_present`.
- Suricata `eve.json` can include deeply nested statistics and alerts.
- Better Suricata features include `suricata_kernel_packets`, `suricata_kernel_drops`, `suricata_alert_count`, `suricata_http_flow_count`, and `suricata_ssh_flow_count`.

## Prototype Encoding Fix

- Keep raw readable strings for the LLM evidence packet.
- Convert raw fields into simple numeric features for Mamba.
- Add missingness flags for important nullable fields.
- Start with security-relevant flags and counts instead of full text embeddings.
- Encode source and event types with fields such as `source_type_zeek_http`, `source_type_host_auth`, and `event_type_http_request`.
- Encode behavior with fields such as `src_ip_internal`, `dst_port_common_http`, `uri_has_cmd`, `uri_has_passwd`, `event_user_created`, and `suricata_alert_count`.
- Encode presence with fields such as `uri_present`, `username_present`, `hash_present`, and `src_ip_present`.

## Stable Facts And Mamba Signal

- Stable identifiers should remain preserved outside the Mamba representation.
- Examples include IP addresses, hostnames, usernames, connection UIDs, file IDs, hashes, timestamps, and source types.
- The same identifier should map to the same encoded identity or stable feature each time it appears.
- Mamba should not be responsible for remembering what an IP literally is.
- Mamba should learn how behavior involving that IP changes across the sequence.
- Mamba's hidden representation can change the contextual meaning of an event based on prior events, but it should not replace the original event facts.
- The architecture should keep two tracks: stable event facts and Mamba learned sequence signal.
- The LLM packet can include both tracks when needed: readable stable facts for grounding and Mamba signal for temporal interpretation.

## Window Design

- Events are grouped into fixed time windows after normalization.
- The current prototype uses 60-second windows.
- This is acceptable because the P025 slice already has timestamps.
- The 60-second choice is still a design assumption, not a final answer.
- Short windows are better for bursts and fast attacks.
- Longer windows are better for slow multi-step behavior.
- Overlapping windows may help preserve context between adjacent periods.
- Window size and overlap should be revisited after the prototype proves the end-to-end path.

## Event Relationship Building

- Relationship building means linking events that appear connected.
- Useful relationship keys include host, source IP, destination IP, username, connection UID, timestamp proximity, file hash, URI, or process identifier.
- Example: a Zeek HTTP request, Suricata alert, and host auth event involving the same host in the same time range.
- Relationship summaries can help the LLM see an incident chain instead of isolated facts.
- This is useful, but not mandatory for the first prototype.
- The current prototype can rely on shared timestamps, source counts, stable identifiers, and readable evidence first.

## Stateful Operation

- Stateful operation is currently handled through app-level stream memory.
- Current stream memory tracks recent windows, previous window ID, embedding L2 norm history, previous Mamba representation summaries, and trend-like fields.
- This is not true recurrent Mamba hidden-state carryover.
- It is sufficient for the current prototype because it gives the LLM recent context across windows.
- True recurrent Mamba state can be explored later if the system moves from replayed windows to a continuous live event stream.

## Mamba Encoder Learning Objective

- The current P025 Mamba encoder path is shape-valid and pipeline-valid.
- It proves that normalized mixed events can become windows, tensors, encoder representations, LLM packets, and Ollama assessments.
- It does not prove that the encoder representation is security-intelligent yet.
- This matters even if Mamba is not used as a classifier.
- If Mamba is used as a classifier, it needs supervised classifier training.
- If Mamba is used only as an encoder, it still needs a learning objective so the representation becomes meaningful.
- Possible future objectives include supervised labels, next-event prediction, reconstruction/anomaly detection, contrastive learning, baseline modeling, or known normal-versus-changed behavior modeling.
- Until that work is done, Mamba diagnostics should be treated as prototype/debug signal, not reliable security evidence.

## Already Scoped Assumptions

- Duplicate evidence is not treated as a major prototype issue.
- Duplicate rows are expected to be handled during shared-shape normalization where possible.
- Timestamp alignment is assumed to exist for testing in the current mixed dataset.
- Different event granularity is not treated as a blocker.
- Volume is not treated as a blocker for the first prototype.
- Mamba labeling is not part of the current direction.
- Formal evaluation is out of scope for the immediate prototype.
- Trust and explainability are not primary constraints right now.
- Current context selection is assumed to be good enough for the prototype.
