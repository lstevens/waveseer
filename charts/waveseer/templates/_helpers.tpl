{{/*
Return the name of the chart
*/}}
{{- define "waveseer.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Return the fullname of the chart
*/}}
{{- define "waveseer.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "waveseer.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{/*
Define standard labels for all resources
*/}}
{{- define "waveseer.labels" -}}
app.kubernetes.io/name: {{ include "waveseer.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
{{- end -}}

{{/*
Define selector labels for Deployment selectors
*/}}
{{- define "waveseer.selectorLabels" -}}
app.kubernetes.io/name: {{ include "waveseer.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
